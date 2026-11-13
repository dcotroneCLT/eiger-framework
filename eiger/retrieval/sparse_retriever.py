"""
SparseRetriever: BM25 (Okapi) lexical retrieval over an in-memory corpus.

This module provides the second concrete BaseRetriever implementation
(alongside DenseRetriever), addressing the "SparseRetriever — BM25 index
via rank-bm25" item on the retrieval roadmap (see eiger/retrieval/README.md
and docs/CLAIM_AND_RESEARCH_QUESTIONS.md §9 — RQ5/H5's hybrid-defense
comparison needs a working sparse baseline before hybrid fusion is possible).

Pipeline position
------------------
    list[Document]  (the full corpus: ground-truth + poisoned)
        │
        ▼  fit(documents)            — tokenize + build BM25Okapi index
    (in-memory index ready)
        │
    query (str)
        │
        ▼  retrieve(query, claim_id, top_k)
    tokenize(query) → bm25.get_scores() → rank → normalize → RetrievalResult

Design decisions
----------------
- **No BaseVectorStore involved, by necessity, not just by choice.**
  BaseVectorStore's interface (create_collection/reset_collection/upsert/
  search) has no method to enumerate every document in a collection — only
  to search by vector. BM25 needs the *entire* tokenized corpus up front to
  compute term frequencies and inverse document frequencies, which a
  vector-similarity search API cannot provide. SparseRetriever therefore
  owns its own corpus state, built explicitly via fit(), rather than
  reading from any BaseVectorStore. This is why fit() is a method on this
  class specifically and not part of the BaseRetriever contract: DenseRetriever
  needs no equivalent step, since the vector store already holds the corpus.
- **Per-query max-normalization, not a fixed formula.** DenseRetriever's
  cosine similarity has a known range ([-1, 1]), so
  DenseRetriever._normalize_score() can use a single linear formula. Okapi
  BM25 scores are unbounded above (and can be negative when a term's IDF is
  negative, i.e. it appears in more than half the corpus), so there is no
  fixed range to map from. Instead, every hit's score is divided by the
  maximum score among that query's own candidate results, so the top hit is
  always 1.0 and the rest are proportionally below it. If the maximum score
  is <= 0 (no candidate has any positive lexical overlap with the query),
  every hit's score is 0.0 rather than dividing by a non-positive number.
- **Empty corpus is not an error.** fit([]) is accepted — mirroring
  BaseDataset.load() potentially returning zero claims — and retrieve()
  then always returns a RetrievalResult with an empty hit list. This also
  covers retrieve() being called before fit() is ever called: internal
  state starts empty, so the behavior is identical to fitting on an empty
  corpus. rank_bm25.BM25Okapi itself raises ZeroDivisionError when
  constructed from an empty corpus (it divides by corpus size to compute
  average document length), so this case is special-cased here rather than
  left to surface as a raw traceback.
- **Tokenization is intentionally simple.** Text is lowercased and split on
  ``\\w+`` (word characters) with no stemming or stopword removal. BM25's
  term-frequency/inverse-document-frequency statistics are what this class
  exists to exercise; a more sophisticated tokenizer is not part of this
  scope and can be swapped in later without changing the public interface.

What this module does NOT do:
  - It does not implement hybrid retrieval (RRF fusion of dense + sparse
    rankings) itself — see eiger/retrieval/hybrid_retriever.py, which
    composes this class and DenseRetriever rather than extending either
    (Sprint 5).
  - It does not persist its index; fit() must be called again for every
    new corpus (this mirrors ExperimentRunner rebuilding the corpus fresh
    on every run() call).
  - It does not fetch documents from a vector store or any other backend.
"""

from __future__ import annotations

import hashlib

import re
from typing import TYPE_CHECKING

from eiger.core.exceptions import RetrievalError
from eiger.core.interfaces import BaseRetriever
from eiger.core.models import Document, RetrievalResult, RetrievedDocument
from eiger.utils.logging import get_logger

if TYPE_CHECKING:
    # Only imported for type hints; the real import happens lazily in
    # fit(), matching the lazy-import pattern used by SentenceTransformerEmbedder
    # and QdrantVectorStore for their own optional heavy dependencies.
    from rank_bm25 import BM25Okapi

log = get_logger(__name__)

# Simple word-character tokenizer: lowercase, split on runs of \w. No
# stemming or stopword removal — see module docstring's "Tokenization"
# design note for why this is intentional.
_TOKEN_PATTERN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens."""
    return _TOKEN_PATTERN.findall(text.lower())


def _tie_break(doc_id: str) -> int:
    """Neutral, deterministic tie-break value for documents with equal BM25 scores."""
    return int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:16], 16)


class SparseRetriever(BaseRetriever):
    """
    BM25 (Okapi) lexical retriever over an explicitly-fitted document corpus.

    Usage::

        retriever = SparseRetriever()
        retriever.fit(corpus.all_documents)  # after CorpusBuilder.build()
        result = retriever.retrieve(
            query="What did the WHO report about 2023 inflation?",
            claim_id="TEST_001",
            top_k=5,
        )

    Unlike DenseRetriever, no embedder or vector store is involved — see
    the module docstring's "No BaseVectorStore involved" design note.
    """

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._bm25: BM25Okapi | None = None

    # ─── Corpus indexing ────────────────────────────────────────────────────────

    def fit(self, documents: list[Document]) -> None:
        """
        Build (or rebuild) the BM25 index from the given corpus.

        Must be called before retrieve() returns any non-empty results;
        calling retrieve() beforehand is not an error (see module
        docstring's "Empty corpus is not an error" design note) but will
        always return zero hits.

        Args:
            documents: Full corpus to index (typically
                       CorpusBuilderResult.all_documents — ground-truth
                       and poisoned documents together). May be empty.

        Raises:
            ImportError: If the optional ``rank-bm25`` dependency is not
                         installed. Raised with an actionable install hint,
                         matching the pattern used by
                         SentenceTransformerEmbedder/QdrantVectorStore for
                         their own optional dependencies.
        """
        self._documents = list(documents)

        if not self._documents:
            self._bm25 = None
            log.info("sparse_retriever.fit_empty_corpus")
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 is required for SparseRetriever. Install it with: "
                "pip install rank-bm25"
            ) from exc

        tokenized_corpus = [_tokenize(doc.text) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized_corpus)
        log.info("sparse_retriever.fitted", n_documents=len(self._documents))

    # ─── BaseRetriever interface ────────────────────────────────────────────────

    def retrieve(self, query: str, claim_id: str, top_k: int) -> RetrievalResult:
        """
        Retrieve the top_k most lexically similar documents for a query.

        Args:
            query:    Natural language query (typically Claim.context_query).
            claim_id: ID of the claim this query belongs to, propagated into
                      the RetrievalResult for downstream traceability.
            top_k:    Maximum number of documents to retrieve.

        Returns:
            RetrievalResult with hits ranked 1..N by descending BM25 score.
            Empty (but valid) if fit() has not been called, or was called
            with an empty corpus.

        Raises:
            RetrievalError: If BM25 scoring fails for any reason.
        """
        if not self._documents:
            log.debug("sparse_retriever.retrieve_empty_corpus", claim_id=claim_id)
            return RetrievalResult(query=query, claim_id=claim_id, hits=[], top_k=top_k)

        assert self._bm25 is not None  # guaranteed by fit() whenever _documents is non-empty

        try:
            scores = self._bm25.get_scores(_tokenize(query))
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            raise RetrievalError(f"BM25 scoring failed for claim '{claim_id}': {exc}") from exc

        # Bug fix (November 2026 review): ties used to be resolved by index order
        # (a stable sort), and CorpusBuilder indexes every ground-truth document
        # BEFORE its poisoned copy. When a query does not contain the token that
        # the manipulation changed, the two documents score exactly the same, so
        # the ground truth always won the tie and PRD@1 for BM25 was ~0 by
        # construction. Ties are now broken by a hash of the doc_id: deterministic
        # and identical on every machine, but independent of document type.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: (-float(scores[i]), _tie_break(self._documents[i].doc_id)),
        )[:top_k]
        max_score = max((float(scores[i]) for i in ranked_indices), default=0.0)

        hits = [
            RetrievedDocument(
                document=self._documents[idx],
                score=self._normalize_score(float(scores[idx]), max_score),
                rank=rank,
            )
            for rank, idx in enumerate(ranked_indices, start=1)
        ]

        log.info("sparse_retriever.retrieved", claim_id=claim_id, n_hits=len(hits), top_k=top_k)
        return RetrievalResult(query=query, claim_id=claim_id, hits=hits, top_k=top_k)

    # ─── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _normalize_score(raw_score: float, max_score: float) -> float:
        """
        Rescale a raw BM25 score into [0, 1] via per-query max-normalization.

        See the module docstring's "Per-query max-normalization" design
        note for the full rationale (BM25 scores have no fixed range,
        unlike DenseRetriever's cosine similarity).

        Args:
            raw_score: This hit's raw BM25 score.
            max_score: The maximum raw BM25 score among this query's own
                       candidate results (already restricted to top_k).

        Returns:
            float in [0.0, 1.0]. 0.0 if max_score is not positive.
        """
        if max_score <= 0.0:
            return 0.0
        normalized = raw_score / max_score
        return max(0.0, min(1.0, normalized))
