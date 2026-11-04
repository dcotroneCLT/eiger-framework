"""
ExperimentRunner: end-to-end orchestration of a single EIBench experiment.

This module is the top-level entry point that ties together every
component built in Sprint 2: corpus construction, vector store ingestion,
dense retrieval, LLM generation, and metric evaluation.

Pipeline position
------------------
    ExperimentConfig + list[Claim]
        │
        ▼  seed_everything(config.seed)                    — reproducibility
        ▼  CorpusBuilder.build(claims)                      — attacks resolved via registry
    CorpusBuilderResult (ground-truth + poisoned Documents)
        │
        ▼  retriever.type == "dense" (default):
        │     IngestionPipeline.ingest(corpus)               — embed + upsert
        │   (vector store populated)
        ▼  retriever.type == "sparse":
        │     SparseRetriever.fit(corpus.all_documents)       — build BM25 index
        │   (vector store untouched — no Qdrant/embedder network calls at all)
        ▼  retriever.type == "hybrid":
        │     IngestionPipeline.ingest(corpus)                — embed + upsert (dense side)
        │     HybridRetriever.fit(corpus.all_documents)       — build BM25 index (sparse side)
        │   (both underlying indexes populated)
        │
        ▼  for each claim:
        │     retriever.retrieve()  → RetrievalResult   (Dense-, Sparse-, or HybridRetriever)
        │     BaseLLM.build_rag_prompt() + generate() → GenerationResult
        │     ["pcs" configured AND retrieval.contains_poisoned]:
        │         a SECOND build_rag_prompt()+generate() call, with poisoned
        │         hits filtered out of context_docs, stashed onto
        │         generation.metadata["counterfactual_answer"] for PCSMetric
        │     [faithfulness_scorer(claim, generation)]  → optional RAGAS-style scores
        │   → EvaluationRecord
        │
        ▼  for each configured metric name:
        │     BaseMetric.compute_batch(records) → per-record scores
        │     BaseMetric.aggregate(scores)       → experiment-level scalar
        │
        ▼  ExperimentResult(records, aggregate_metrics, git_commit, environment, ...)
        ▼  write to {config.output_dir}/results.json   [if save=True]

Design decisions
----------------
- **Dependency injection for infrastructure, string config for provenance**:
  ExperimentRunner is constructed with already-instantiated ``embedder``,
  ``vector_store``, and ``llm`` objects rather than building them from
  ``config.retriever.embedder`` / ``config.retriever.vector_store`` /
  ``config.llm.backend`` internally. Those config string fields exist for
  provenance and reproducibility (they are serialized into every result
  file via ``ExperimentConfig``), but there is no embedder/vector-store/LLM
  *factory* in EIGER yet — the caller (a CLI entry point or notebook) is
  responsible for constructing objects that match the config and injecting
  them here. This mirrors the DI pattern already used by DenseRetriever and
  IngestionPipeline and keeps ExperimentRunner trivially testable with mocks.
- **Attacks and metrics resolved via the existing registries**: attack names
  in ``config.attacks`` are resolved through ``eiger.attacks.get_attack``,
  and metric names in ``config.metrics`` through ``eiger.metrics.get_metric``.
  ExperimentRunner does not maintain its own registry.
- **``run()`` takes claims, not a dataset name**: ``run()`` accepts an
  already-loaded ``list[Claim]`` directly rather than a dataset name or
  ``BaseDataset`` instance. This was true back in Sprint 2 (when
  ``eiger/datasets/`` did not exist yet) and remains true now that Sprint 3
  added five real ``BaseDataset`` loaders (see ``eiger.datasets``) — loading
  claims from a named dataset is the CLI's job (``eiger/__main__.py``'s
  ``_build_dataset`` + ``dataset.load(...)``), not this class's.
- **FFR requires an external faithfulness signal — this is NOT computed
  here**: ``FFRMetric`` reads ``EvaluationRecord.faithfulness_score`` and
  ``.factual_correctness_score``, which are read from
  ``record.metrics["ragas_faithfulness"]`` / ``["ragas_answer_correctness"]``.
  ExperimentRunner exposes a ``faithfulness_scorer`` hook — a callable that
  receives ``(claim, generation)`` and returns a dict to merge into the
  record's metrics — so any scorer can be plugged in without changing this
  class. Two are available (selected by the CLI via
  ``ExperimentConfig.faithfulness_scorer``, see ``eiger/__main__.py``):
  ``eiger.metrics.EmbeddingFaithfulnessScorer``, a ready-to-use, LLM-judge-
  free cosine-similarity *proxy* (see its module docstring for exactly what
  it does and does not capture — it must be reported as a proxy, not as
  RAGAS, in any published result); and, as of Sprint 5,
  ``eiger.metrics.RAGASFaithfulnessScorer``, a real RAGAS integration using
  an Ollama-served LLM as judge via ``LangchainLLMWrapper`` (see its own
  module docstring for the exact pinned dependency versions this required —
  newer ``ragas`` releases are broken at import time — and for what has and
  has not been verified about it). If "ffr" is configured without any
  scorer at all, a warning is logged once per run: the resulting FFR values
  would silently be 0.0 for every record (faithfulness/correctness default
  to 0.0), which is NOT a valid experimental measurement and must not be
  reported as one.
- **PCS's counterfactual generation lives in `_evaluate_claim`, not in a
  metric or a hook**: unlike `faithfulness_scorer` (a pure scoring callable
  with no LLM access), `PCSMetric` needs an actual second LLM generation —
  same query, context with poisoned hits removed — which only
  `ExperimentRunner` can trigger (`BaseMetric.compute()` runs later, in a
  read-only pass over already-built records, with no access to `self.llm`).
  So `_evaluate_claim` runs this second generation itself, but ONLY when
  `"pcs"` is in `config.metrics` AND the retrieval actually contains a
  poisoned hit (see `_add_counterfactual_generation`) — otherwise every
  claim in every experiment would pay for an extra LLM call it never uses.
  The result is stashed on `generation.metadata["counterfactual_answer"]`
  (and `["counterfactual_context_docs"]`) rather than a new
  `EvaluationRecord` field, so the schema is unchanged and `PCSMetric`
  simply reads it back out of the record it's given.
- **Retriever choice branches in two places, not via a factory**: ``__init__``
  picks ``DenseRetriever``, ``SparseRetriever``, or ``HybridRetriever`` (a
  composition of both — see ``eiger/retrieval/hybrid_retriever.py``) based on
  ``config.retriever.type``, and ``run()`` correspondingly calls
  ``ingestion_pipeline.ingest(corpus)`` (dense side), ``retriever.fit(...)``
  (sparse side), or **both** for ``"hybrid"`` — see
  ``eiger/retrieval/sparse_retriever.py``'s module docstring for why
  ``fit()`` cannot be folded into a common ``BaseRetriever`` method.
  ``HybridRetriever.fit()`` is a thin passthrough to its internal
  ``SparseRetriever.fit()``, so ``run()`` can call ``retriever.fit(...)``
  uniformly for both ``"sparse"`` and ``"hybrid"`` without an `isinstance`
  check.
- **Fail loud, no per-claim error swallowing**: a single claim's retrieval
  or generation failure aborts the whole run (RetrievalError/GenerationError
  propagate unchanged). Silently skipping failed claims would silently bias
  aggregate metrics like FFR — unacceptable for a benchmark whose entire
  purpose is measuring that exact failure mode.
- **Reproducibility fields filled here, not by the caller**: ``git_commit``
  (via ``git rev-parse HEAD``) and ``environment`` (Python/platform info)
  are populated by ExperimentRunner itself, matching the responsibility
  described in ``ExperimentResult``'s own docstring.
- **Result filename is exactly ``results.json``**: per ``ExperimentResult``'s
  docstring ("Serialized to JSON in output_dir/results.json"). Callers who
  run multiple experiments should therefore give each one a distinct
  ``config.output_dir`` (e.g. incorporating ``experiment_id``) to avoid
  overwriting a previous run's results.

What this module does NOT do:
  - It does not load claims from a named dataset (see "No dataset loader" above).
  - It does not compute RAGAS faithfulness/answer-correctness scores itself.
  - It does not construct embedder/vector_store/llm instances from config.
  - It does not retry failed claims or continue past the first error.
"""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Callable
from pathlib import Path

from eiger.attacks import get_attack
from eiger.core.interfaces import BaseEmbedder, BaseLLM, BaseMetric, BaseVectorStore
from eiger.core.models import (
    Claim,
    EvaluationRecord,
    ExperimentConfig,
    ExperimentResult,
    GenerationResult,
    RetrievalResult,
)
from eiger.ingestion import CorpusBuilder, CorpusBuilderResult, IngestionPipeline
from eiger.metrics import get_metric
from eiger.metrics.pcs import PCSMetric
from eiger.retrieval import DenseRetriever, HybridRetriever, SparseRetriever
from eiger.utils.logging import get_logger
from eiger.utils.seeding import seed_everything

log = get_logger(__name__)

# Repository root, resolved relative to this file (eiger/experiments/runner.py
# -> eiger/ -> repo root), so `git rev-parse HEAD` works regardless of the
# caller's current working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Result files are always written under this exact name within
# config.output_dir, per ExperimentResult's own docstring contract.
_RESULT_FILENAME = "results.json"


class ExperimentRunner:
    """
    Orchestrates a full EIBench experiment: corpus → ingestion → retrieval
    → generation → metrics → ExperimentResult.

    Args:
        config:       Validated experiment configuration.
        embedder:     Embedder instance shared by ingestion and retrieval —
                      must be the same model for both (see BaseEmbedder).
        vector_store: Vector store instance shared by ingestion and retrieval.
        llm:          LLM backend used for RAG generation.
        faithfulness_scorer: Optional callable ``(claim, generation) -> dict``
                      that computes external faithfulness/correctness signals
                      (e.g. via RAGAS) to merge into each EvaluationRecord's
                      metrics before FFR (and any other metric relying on
                      them) is computed. Expected keys: "ragas_faithfulness",
                      "ragas_answer_correctness". If omitted and "ffr" is in
                      ``config.metrics``, a warning is logged once per run.

    Example::

        runner = ExperimentRunner(
            config=experiment_config,
            embedder=SentenceTransformerEmbedder(),
            vector_store=QdrantVectorStore(),
            llm=OllamaLLM(model_name=experiment_config.llm.model),
        )
        result = runner.run(claims)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        llm: BaseLLM,
        faithfulness_scorer: Callable[[Claim, GenerationResult], dict[str, float]]
        | None = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.faithfulness_scorer = faithfulness_scorer

        self.collection = config.retriever.collection_name
        self.top_k = config.retriever.top_k
        self.retriever_type = config.retriever.type

        # Both components share the same embedder/vector_store/collection,
        # which is the correctness requirement documented on BaseEmbedder
        # and BaseRetriever. Constructed unconditionally even for sparse
        # retrieval (harmless — both are lazy, no network I/O happens here)
        # so that switching retriever.type in config never requires
        # touching this constructor's call site.
        self.ingestion_pipeline = IngestionPipeline(
            embedder=embedder, vector_store=vector_store, collection=self.collection
        )

        # SparseRetriever owns its own in-memory corpus (see its module
        # docstring for why it cannot read from BaseVectorStore) and is
        # populated via fit() in run(), once the corpus exists — unlike
        # DenseRetriever, which is fully usable as soon as the vector store
        # is populated by ingestion_pipeline.ingest(). HybridRetriever needs
        # both: its internal DenseRetriever is ready once the vector store
        # is populated, and its internal SparseRetriever needs fit() — see
        # run()'s "hybrid" branch below.
        self.retriever: DenseRetriever | SparseRetriever | HybridRetriever
        if self.retriever_type == "sparse":
            self.retriever = SparseRetriever()
        elif self.retriever_type == "hybrid":
            self.retriever = HybridRetriever(
                dense=DenseRetriever(
                    embedder=embedder, vector_store=vector_store, collection=self.collection
                ),
                sparse=SparseRetriever(),
            )
        else:
            self.retriever = DenseRetriever(
                embedder=embedder, vector_store=vector_store, collection=self.collection
            )

    # ─── Public API ────────────────────────────────────────────────────────────

    def run(self, claims: list[Claim], save: bool = True) -> ExperimentResult:
        """
        Run the full experiment pipeline for a list of claims.

        Args:
            claims: Already-loaded claims to build the corpus from and
                    evaluate. May be empty (produces an empty, but valid,
                    ExperimentResult).
            save:   If True (default), write the result to
                    ``{config.output_dir}/results.json`` via save_result().

        Returns:
            The completed ExperimentResult (always returned in memory,
            regardless of ``save``).

        Raises:
            AttackNotFoundError:  If config.attacks references an
                                   unregistered attack name.
            MetricNotFoundError:  If config.metrics references an
                                   unregistered metric name.
            IngestionError:       If embedding or upserting the corpus fails.
            RetrievalError:       If retrieval fails for any claim.
            GenerationError:      If LLM generation fails for any claim.
        """
        log.info(
            "experiment.start",
            experiment_id=self.config.experiment_id,
            n_claims=len(claims),
            metrics=self.config.metrics,
        )

        # Seed every global RNG (Python/numpy/torch) before any stochastic
        # operation (attack application, model sampling) for reproducibility.
        seed_everything(self.config.seed)

        self._warn_if_ffr_unsupported()

        corpus = self._build_corpus(claims)
        if self.retriever_type == "sparse":
            # SparseRetriever needs the explicit document list (see its
            # module docstring); it does not touch the vector store at all,
            # so ingestion_pipeline.ingest() is skipped entirely — a
            # sparse-only run does not require Qdrant to be reachable.
            self.retriever.fit(corpus.all_documents)  # type: ignore[union-attr]
        elif self.retriever_type == "hybrid":
            # Hybrid needs both indexes populated: the dense side reads from
            # the vector store (like a pure "dense" run), and the sparse
            # side needs the explicit document list (like a pure "sparse"
            # run) — HybridRetriever.fit() is a passthrough to its internal
            # SparseRetriever.fit(), so this call is uniform with the
            # "sparse" branch above.
            self.ingestion_pipeline.ingest(corpus)
            self.retriever.fit(corpus.all_documents)  # type: ignore[union-attr]
        else:
            self.ingestion_pipeline.ingest(corpus)

        records = [self._evaluate_claim(claim) for claim in claims]

        aggregate_metrics = self._compute_metrics(records)

        result = ExperimentResult(
            experiment_id=self.config.experiment_id,
            config_hash=self.config.config_hash,
            git_commit=self._get_git_commit(),
            config=self.config,
            records=records,
            aggregate_metrics=aggregate_metrics,
            environment=self._capture_environment(),
        )

        if save:
            self.save_result(result)

        log.info(
            "experiment.complete",
            experiment_id=self.config.experiment_id,
            n_records=len(records),
            aggregate_metrics=aggregate_metrics,
        )
        return result

    def save_result(self, result: ExperimentResult) -> Path:
        """
        Write an ExperimentResult to ``{config.output_dir}/results.json``.

        Creates ``config.output_dir`` (and any parent directories) if it
        does not already exist. Exposed as a public method so a previously
        computed result can be (re-)persisted independently of run().

        Args:
            result: The ExperimentResult to serialize.

        Returns:
            Path to the written JSON file.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / _RESULT_FILENAME
        path.write_text(result.to_json())
        log.info("experiment.saved", path=str(path))
        return path

    # ─── Internal helpers — corpus & ingestion ────────────────────────────────

    def _build_corpus(self, claims: list[Claim]) -> CorpusBuilderResult:
        """
        Resolve configured attacks by name and build the mixed corpus.

        Args:
            claims: Claims to build ground-truth and poisoned documents from.

        Returns:
            CorpusBuilderResult with ground-truth and poisoned documents.

        Raises:
            AttackNotFoundError: If any config.attacks entry names an
                                  unregistered attack.
            ConfigurationError:  If config.attacks includes an attack flagged
                                  excluded_from_benchmark (currently
                                  AttributionSwitchAttack/M03,
                                  CherryPickingAttack/M05 — see
                                  CorpusBuilder's own docstring) and
                                  config.allow_non_benchmark_attacks is False.
        """
        # get_attack() instantiates a fresh attack object per call, matching
        # CorpusBuilder's expected list[tuple[BaseAttack, AttackConfig]].
        attacks = [(get_attack(attack_cfg.name), attack_cfg) for attack_cfg in self.config.attacks]
        return CorpusBuilder(
            attacks=attacks,
            seed=self.config.seed,
            allow_non_benchmark_attacks=self.config.allow_non_benchmark_attacks,
        ).build(claims)

    # ─── Internal helpers — per-claim retrieval & generation ──────────────────

    def _evaluate_claim(self, claim: Claim) -> EvaluationRecord:
        """
        Retrieve context, generate an answer, and assemble an EvaluationRecord
        for a single claim.

        Args:
            claim: The claim to evaluate.

        Returns:
            EvaluationRecord with retrieval + generation populated, and
            metrics pre-seeded with any faithfulness_scorer output.

        Raises:
            RetrievalError:  If retrieval fails.
            GenerationError: If LLM generation fails.
        """
        retrieval = self.retriever.retrieve(
            query=claim.context_query, claim_id=claim.claim_id, top_k=self.top_k
        )
        context_docs = [hit.document.text for hit in retrieval.hits]

        prompt = self.llm.build_rag_prompt(claim.context_query, context_docs)
        # Explicitly pass the configured temperature/max_tokens on every call
        # so the *actual* generation parameters always match config.llm,
        # regardless of what defaults the injected llm instance happens to
        # have been constructed with — critical for config_hash-based
        # reproducibility guarantees.
        answer = self.llm.generate(
            prompt,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )

        generation = GenerationResult(
            claim_id=claim.claim_id,
            query=claim.context_query,
            context_docs=context_docs,
            answer=answer,
            model_name=self.llm.model_name,
            metadata={
                "temperature": self.config.llm.temperature,
                "max_tokens": self.config.llm.max_tokens,
                # Read by SourceIntegrityMetric (and any metric comparing
                # retrieved context against ground truth).
                "ground_truth": claim.original_fact,
            },
        )

        if "pcs" in self.config.metrics and retrieval.contains_poisoned:
            self._add_counterfactual_generation(claim, retrieval, generation)

        metrics: dict[str, float] = {}
        if self.faithfulness_scorer is not None:
            metrics.update(self.faithfulness_scorer(claim, generation))

        return EvaluationRecord(
            claim_id=claim.claim_id,
            generation=generation,
            retrieval=retrieval,
            metrics=metrics,
        )

    def _add_counterfactual_generation(
        self, claim: Claim, retrieval: RetrievalResult, generation: GenerationResult
    ) -> None:
        """
        Run a second, counterfactual generation with poisoned context removed,
        and stash it on ``generation.metadata`` for ``PCSMetric`` to read.

        Only called from ``_evaluate_claim`` when "pcs" is configured AND the
        retrieval actually contains at least one poisoned document — running
        a second LLM call for every claim regardless of whether there is any
        poisoned context to remove would be pure overhead for a metric that
        specifically measures sensitivity *to poisoned context*.

        Mutates ``generation.metadata`` in place (GenerationResult is not
        frozen) rather than adding a new EvaluationRecord field, mirroring
        how ``temperature``/``max_tokens`` are already stored there — this
        keeps ``EvaluationRecord``'s schema unchanged and the counterfactual
        answer travels with the generation it is a counterfactual *of*.

        Args:
            claim:      Source claim (for context_query — same query is reused).
            retrieval:  The original RetrievalResult (already known to contain
                        at least one poisoned hit).
            generation: The real GenerationResult, mutated in place: adds
                        "counterfactual_answer" (str) and
                        "counterfactual_context_docs" (list[str], possibly
                        empty if every retrieved hit was poisoned) keys to
                        ``generation.metadata``.
        """
        clean_docs = [
            hit.document.text for hit in retrieval.hits if hit.document.doc_type != "poisoned"
        ]
        # build_rag_prompt/generate handle an empty context_docs list without
        # special-casing here — a fully-poisoned top-k is itself a valid
        # experimental condition (PCSMetric can report it via the
        # "counterfactual_context_docs" metadata rather than this method
        # silently substituting or skipping it).
        counterfactual_prompt = self.llm.build_rag_prompt(claim.context_query, clean_docs)
        counterfactual_answer = self.llm.generate(
            counterfactual_prompt,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
        )
        generation.metadata["counterfactual_answer"] = counterfactual_answer
        generation.metadata["counterfactual_context_docs"] = clean_docs

    # ─── Internal helpers — metrics ────────────────────────────────────────────

    def _warn_if_ffr_unsupported(self) -> None:
        """
        Log a warning if FFR is configured without a faithfulness_scorer.

        Without pre-computed faithfulness/answer-correctness scores, every
        record's faithfulness_score and factual_correctness_score default to
        0.0 (see EvaluationRecord), which makes FFRMetric.compute() always
        return 0.0. That is silently wrong, not a real measurement, so this
        is surfaced loudly (once, at the start of the run) rather than left
        for the user to discover in a suspiciously-flat results file.
        """
        if "ffr" in self.config.metrics and self.faithfulness_scorer is None:
            log.warning(
                "experiment.ffr_without_faithfulness_scorer",
                message=(
                    "FFR is configured but no faithfulness_scorer was provided. "
                    "faithfulness_score/factual_correctness_score will default "
                    "to 0.0 for every record, so the resulting FFR is NOT a "
                    "valid measurement — it will trivially be 0.0."
                ),
            )

    def _compute_metrics(self, records: list[EvaluationRecord]) -> dict[str, float]:
        """
        Compute every configured metric over all records.

        For each metric name in config.metrics: resolves the metric via the
        registry, computes a per-record score for every record (writing it
        back into ``record.metrics`` so the persisted result is
        self-contained), and aggregates to an experiment-level scalar.

        Args:
            records: Completed evaluation records (retrieval + generation).

        Returns:
            Dict mapping metric name to its experiment-level aggregate value.

        Raises:
            MetricNotFoundError: If config.metrics names an unregistered metric.
        """
        aggregate_metrics: dict[str, float] = {}
        for metric_name in self.config.metrics:
            metric = self._resolve_metric(metric_name)
            scores = metric.compute_batch(records)
            # Write each per-record score back into the record itself so the
            # serialized ExperimentResult is self-contained (no need to
            # re-run metrics to inspect per-claim values).
            for record, score in zip(records, scores, strict=True):
                record.metrics[score.metric_name] = score.value
            aggregate_metrics[metric_name] = metric.aggregate(scores)
        return aggregate_metrics

    def _resolve_metric(self, metric_name: str) -> BaseMetric:
        """
        Resolve a configured metric name to an instance, special-casing "pcs".

        Every other metric is a zero-argument class in the registry (see
        ``eiger.metrics.registry``'s own design note: ``get_metric`` always
        instantiates with no arguments, by design, so metrics never carry
        shared mutable state across runs). ``PCSMetric`` is the one
        exception — it genuinely needs an embedder to compare the real and
        counterfactual answers (see ``eiger/metrics/pcs.py``) — so
        ``get_metric("pcs")`` would raise a ``TypeError`` if called directly.
        ``ExperimentRunner`` already holds an ``embedder`` (shared with
        ingestion/retrieval), so it constructs ``PCSMetric`` directly here
        instead of going through the no-argument registry path.

        Args:
            metric_name: A name from ``config.metrics``.

        Returns:
            A ready-to-use ``BaseMetric`` instance.

        Raises:
            MetricNotFoundError: If ``metric_name`` is neither "pcs" nor a
                                  name registered in ``eiger.metrics.registry``.
        """
        if metric_name == "pcs":
            return PCSMetric(embedder=self.embedder)
        return get_metric(metric_name)

    # ─── Internal helpers — reproducibility metadata ──────────────────────────

    @staticmethod
    def _get_git_commit() -> str:
        """
        Return the current git commit SHA of the EIGER repository.

        Runs `git rev-parse HEAD` with cwd set to the repo root (resolved
        relative to this file), so the result does not depend on the
        caller's working directory.

        Returns:
            The full commit SHA, or "unknown" if git is unavailable, this
            is not a git repository, or the command fails for any reason.
        """
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return completed.stdout.strip()
        except Exception as exc:  # noqa: BLE001 - absence of git is expected outside a repo
            log.debug("experiment.git_commit_unavailable", error=str(exc))
            return "unknown"

    @staticmethod
    def _capture_environment() -> dict[str, str]:
        """
        Capture basic environment metadata for cross-machine debugging.

        Returns:
            Dict with "python_version" and "platform" keys, so results
            produced on different machines can be compared for
            environment-induced discrepancies.
        """
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }
