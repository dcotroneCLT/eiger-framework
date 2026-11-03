"""
RAGASFaithfulnessScorer: real RAGAS-based faithfulness_scorer, using an
Ollama-served LLM as the judge.

Background — why this exists
-----------------------------
``eiger.metrics.EmbeddingFaithfulnessScorer`` (heuristic_scorer.py) has,
since Sprint 2, been an explicitly-labelled *proxy* for the faithfulness/
answer-correctness signals FFRMetric needs — a cosine-similarity shortcut
adopted specifically because a real RAGAS integration was, at the time, an
unverified "substantial undertaking" (see that module's own docstring).
This module is that undertaking, completed and verified in Sprint 5.

Verified, pinned dependency versions — READ BEFORE CHANGING
--------------------------------------------------------------
This is not an arbitrary pin. As of this writing, installing the *latest*
``ragas`` (0.4.x) or even ``ragas`` 0.3.x into a clean environment fails at
import time::

    ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'

This happens because ``ragas.llms.base`` unconditionally imports
``ChatVertexAI`` from ``langchain_community`` at module load, but
``langchain-community`` has been "sunset" (see
github.com/langchain-ai/langchain-community/issues/674) and, as of its
0.4.x releases, no longer bundles that submodule at all (it moved to a
separate ``langchain-google-vertexai`` package that does not provide it as
a drop-in replacement for a `langchain-community` import path). This
failure was reproduced directly (in a clean venv, `pip install ragas`)
while building this module — it is not a hypothetical concern copied from
an issue tracker.

The one verified-working combination, confirmed by actually importing
``ragas.metrics.Faithfulness``/``AnswerCorrectness`` and
``ragas.llms.LangchainLLMWrapper`` in a clean virtualenv, is::

    ragas==0.2.15
    langchain-community==0.3.19
    langchain-ollama==0.2.3

These are the exact versions pinned in ``pyproject.toml``'s ``ragas``
optional-dependency group. Do not relax these pins to "latest" without
re-verifying the import chain above in a clean environment first — this
exact failure mode is silent until someone actually tries to import the
package, which is precisely how it went unnoticed for as long as it did.

What is NOT verified
----------------------
Import-time correctness and the exact API shape used below (constructor
signatures, required ``SingleTurnSample`` columns) were verified directly.
What was NOT verified — because it requires a live Ollama server, which is
outside this sandbox's reach — is end-to-end scoring quality: whether
Ollama-as-judge actually produces reliable faithfulness/correctness scores
for EIBench's specific claims. This has documented upstream reliability
issues (see explodinggradients/ragas issues #1120, #1246, already flagged
in heuristic_scorer.py's own docstring) that a local, non-frontier model
like Llama 3.1 8B is likely to make worse, not better. Before reporting any
number produced by this scorer, run a small manual spot-check (a handful of
claims where you already know the right answer) and report results as
"FFR (RAGAS, <model> judge)" — never as unqualified "FFR" — so any
downstream reader knows exactly which judge produced the number.

Swapping this in for (or alongside) EmbeddingFaithfulnessScorer requires no
change to ExperimentRunner: any callable with signature
``(Claim, GenerationResult) -> dict[str, float]`` works via the
``faithfulness_scorer`` constructor argument — see
``eiger.experiments.runner``. The CLI (``eiger/__main__.py``) selects
between them via ``ExperimentConfig.faithfulness_scorer``
(``"embedding"`` | ``"ragas"`` | ``"none"``).

Score semantics
---------------
- ``ragas_faithfulness``: RAGAS's ``Faithfulness`` metric — decomposes the
  answer into atomic statements and asks the judge LLM whether each is
  entailed by the retrieved context, returning the fraction that are.
  Unlike the embedding proxy, this is a real (if judge-quality-bounded)
  entailment check, not a similarity heuristic.
- ``ragas_answer_correctness``: RAGAS's ``AnswerCorrectness`` metric — a
  weighted combination of factual overlap (judged by the LLM, comparing
  the answer against ``claim.original_fact`` as the reference) and
  semantic similarity (via the injected embedder).

Both keys populate exactly the fields ``FFRMetric`` reads, identically to
``EmbeddingFaithfulnessScorer``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from eiger.core.interfaces import BaseEmbedder
from eiger.core.models import Claim, GenerationResult
from eiger.utils.logging import get_logger

if TYPE_CHECKING:
    # Only imported for type hints; the real imports happen lazily in
    # __init__ (see this module's docstring for why the exact versions
    # matter), matching the lazy-import pattern used by
    # SentenceTransformerEmbedder, QdrantVectorStore, OllamaLLM, and
    # SparseRetriever for their own optional/heavy dependencies.
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import AnswerCorrectness, Faithfulness

log = get_logger(__name__)

# The exact combination verified to import cleanly — see module docstring.
_VERIFIED_VERSIONS = (
    "ragas==0.2.15 langchain-community==0.3.19 langchain-ollama==0.2.3"
)

# AnswerCorrectness weights [factual_overlap, semantic_similarity], pinned
# here to the RAGAS 0.2.15 defaults (see module docstring).
_ANSWER_CORRECTNESS_WEIGHTS = [0.75, 0.25]


class _EmbedderAsLangchainEmbeddings:
    """
    Duck-typed adapter from ``BaseEmbedder`` to langchain's ``Embeddings``
    interface, so an existing ``SentenceTransformerEmbedder`` instance can
    be reused as RAGAS's ``AnswerCorrectness`` similarity embedder instead
    of introducing a second, separate embedding backend.

    Not a formal subclass of ``langchain_core.embeddings.Embeddings``
    (which would require importing it at module load time, defeating the
    lazy-import design) — verified directly that
    ``ragas.embeddings.LangchainEmbeddingsWrapper`` accepts any object
    exposing this method set without an isinstance check.
    """

    def __init__(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.encode([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        # SentenceTransformerEmbedder.encode() is CPU-bound and synchronous;
        # there is no real async work to do here, so the sync path is
        # called directly rather than via a thread/executor.
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


class RAGASFaithfulnessScorer:
    """
    Real RAGAS-based faithfulness/answer-correctness scorer, using an
    Ollama-served LLM as the judge via ``ragas.llms.LangchainLLMWrapper``.

    Args:
        embedder:    Any BaseEmbedder implementation, reused as
                     AnswerCorrectness's semantic-similarity embedder (see
                     ``_EmbedderAsLangchainEmbeddings``). Using the same
                     instance already shared for retrieval/ingestion is a
                     reasonable default, but not required.
        model_name:  Ollama model to use as judge. Defaults to the same
                     model EIGER uses for generation, but a judge model
                     does not have to match the generation model.
        host:        Ollama server host.
        port:        Ollama server port.
        temperature: Judge sampling temperature. Defaults to 0.0 for
                     reproducibility, matching the rest of EIGER's
                     determinism conventions.

    Raises:
        ImportError: If ``ragas``/``langchain-ollama`` (pinned combination
                     — see module docstring) are not installed. Raised with
                     an actionable install hint, matching the pattern used
                     by every other optional-dependency class in EIGER.

    Example::

        scorer = RAGASFaithfulnessScorer(
            embedder=SentenceTransformerEmbedder(),
            model_name="llama3.1:8b",
        )
        runner = ExperimentRunner(
            config=config,
            embedder=embedder,
            vector_store=vector_store,
            llm=llm,
            faithfulness_scorer=scorer,
        )
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        model_name: str = "llama3.1:8b",
        host: str = "localhost",
        port: int = 11434,
        temperature: float = 0.0,
    ) -> None:
        try:
            from langchain_ollama import ChatOllama
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import AnswerCorrectness, AnswerSimilarity, Faithfulness
        except ImportError as exc:
            raise ImportError(
                "RAGASFaithfulnessScorer requires the pinned 'ragas' extra "
                f"({_VERIFIED_VERSIONS}). Install it with: "
                f"pip install {_VERIFIED_VERSIONS} "
                "— see this module's docstring for why these exact "
                "versions are required (newer releases are currently "
                "broken at import time)."
            ) from exc

        # Logged once at construction (not per-call), mirroring
        # EmbeddingFaithfulnessScorer's own "impossible to miss, not spammy"
        # convention — this is a real judge, not a proxy, but Ollama-as-
        # judge reliability is still unvalidated for EIBench's claims (see
        # module docstring).
        log.warning(
            "ragas_scorer.judge_in_use",
            message=(
                f"RAGASFaithfulnessScorer is using Ollama model "
                f"'{model_name}' as an LLM judge. This is real RAGAS "
                "scoring, not a proxy, but has NOT been independently "
                "validated against human judgments for EIBench's claims. "
                "Report results as 'FFR (RAGAS, "
                f"{model_name} judge)', not unqualified 'FFR'."
            ),
        )

        chat_llm = ChatOllama(
            model=model_name,
            base_url=f"http://{host}:{port}",
            temperature=temperature,
        )
        wrapped_llm = LangchainLLMWrapper(chat_llm)
        wrapped_embeddings = LangchainEmbeddingsWrapper(
            _EmbedderAsLangchainEmbeddings(embedder)
        )

        self._faithfulness: Faithfulness = Faithfulness(llm=wrapped_llm)
        self._answer_correctness: AnswerCorrectness = AnswerCorrectness(
            llm=wrapped_llm, embeddings=wrapped_embeddings
        )
        # RAGAS 0.2.15 creates AnswerSimilarity only in __post_init__, and in
        # our integration test that step did not populate it ("AnswerSimilarity
        # must be set"). Build it explicitly, with the same embedder and fixed
        # weights, so the scorer does not depend on library post-init behaviour.
        self._answer_correctness.weights = list(_ANSWER_CORRECTNESS_WEIGHTS)
        self._answer_correctness.answer_similarity = AnswerSimilarity(
            embeddings=wrapped_embeddings
        )

    def __call__(self, claim: Claim, generation: GenerationResult) -> dict[str, float]:
        """
        Compute real faithfulness/correctness scores for one generation.

        Args:
            claim:      Source claim; provides ``original_fact`` as the
                        reference for AnswerCorrectness.
            generation: RAG generation result (query + answer + context_docs).

        Returns:
            Dict with "ragas_faithfulness" and "ragas_answer_correctness"
            keys, each in [0.0, 1.0]. Both default to 0.0 — without
            invoking the judge LLM at all — if the answer or context is
            empty/blank, mirroring EmbeddingFaithfulnessScorer's guard
            (nothing meaningful to judge, and avoids wasting an LLM call).
        """
        answer = generation.answer
        context_docs = generation.context_docs

        if not answer.strip() or not any(doc.strip() for doc in context_docs):
            return {"ragas_faithfulness": 0.0, "ragas_answer_correctness": 0.0}

        from ragas.dataset_schema import SingleTurnSample  # lazy, see __init__

        sample = SingleTurnSample(
            user_input=generation.query,
            response=answer,
            retrieved_contexts=context_docs,
            reference=claim.original_fact,
        )

        faithfulness_score, correctness_score = asyncio.run(self._score_both(sample))
        return {
            "ragas_faithfulness": faithfulness_score,
            "ragas_answer_correctness": correctness_score,
        }

    # ─── Internal helpers ─────────────────────────────────────────────────────

    async def _score_both(self, sample: SingleTurnSample) -> tuple[float, float]:
        """
        Run both metrics' async scoring concurrently (they use independent
        ragas.metrics instances, so there is no shared state to race on).

        Args:
            sample: A fully-populated SingleTurnSample.

        Returns:
            (faithfulness_score, answer_correctness_score) tuple.
        """
        faithfulness_score, correctness_score = await asyncio.gather(
            self._faithfulness.single_turn_ascore(sample),
            self._answer_correctness.single_turn_ascore(sample),
        )
        return faithfulness_score, correctness_score
