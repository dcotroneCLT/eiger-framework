"""
Unit tests for ExperimentRunner (eiger.experiments.runner).

Tests verify:
  - __init__ stores config/embedder/vector_store/llm/faithfulness_scorer
  - collection and top_k are derived from config.retriever
  - ingestion_pipeline and retriever are built with the shared embedder/
    vector_store/collection
  - run() resolves configured attacks via the real attack registry and
    builds ground-truth + poisoned documents accordingly
  - run() raises AttackNotFoundError for an unregistered attack name
  - run() ingests the corpus (reset_collection + upsert called)
  - run() retrieves context and calls build_rag_prompt()/generate() per claim
    with the configured temperature/max_tokens
  - run() assembles GenerationResult/EvaluationRecord correctly
  - run() computes configured metrics via the real metric registry and
    writes per-record scores back into record.metrics
  - run() raises MetricNotFoundError for an unregistered metric name
  - the faithfulness_scorer hook's output is merged into record.metrics
    before FFR is computed
  - a warning is logged when "ffr" is configured without a faithfulness_scorer
  - _get_git_commit() returns a SHA on success and "unknown" on failure
  - _capture_environment() returns python_version/platform keys
  - run(save=True) writes {output_dir}/results.json; run(save=False) does not
  - an empty claims list produces a valid, empty ExperimentResult

What these tests do NOT cover:
  - A real running Ollama/Qdrant/sentence-transformers stack (covered by
    each component's own unit tests and by integration tests).
  - SourceIntegrityMetric's NLI inference (covered in test_source_integrity.py;
    here it is exercised only via its documented transformers-missing fallback).
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eiger.core.exceptions import AttackNotFoundError, MetricNotFoundError
from eiger.core.interfaces import BaseEmbedder, BaseLLM, BaseVectorStore
from eiger.core.models import (
    AttackConfig,
    Claim,
    DatasetConfig,
    ExperimentConfig,
    ExperimentResult,
    GenerationResult,
    LLMConfig,
    RetrieverConfig,
)
from eiger.experiments.runner import ExperimentRunner
from eiger.retrieval import DenseRetriever, HybridRetriever, SparseRetriever

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _silence_logger() -> Iterator[None]:
    """
    Patch the module-level structlog logger for every test in this file.

    Matches the pattern used in test_retriever.py / test_pipeline.py /
    test_ollama.py to avoid a structlog version quirk (PrintLogger has no
    .name) that surfaces whenever configure_logging() has not been called.
    """
    with patch("eiger.experiments.runner.log"):
        yield


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_claim(claim_id: str = "C1", text: str = "Inflation rose to 3.5% in 2023.") -> Claim:
    """Return a minimal Claim for runner tests."""
    return Claim(
        claim_id=claim_id,
        original_fact=text,
        context_query=f"What happened in {claim_id}?",
        source_dataset="test_fixture",
    )


def _make_config(
    tmp_path: Path,
    attacks: list[AttackConfig] | None = None,
    metrics: list[str] | None = None,
    top_k: int = 5,
    collection_name: str = "eiger_corpus",
    temperature: float = 0.0,
    max_tokens: int = 512,
    retriever_type: str = "dense",
    allow_non_benchmark_attacks: bool = False,
) -> ExperimentConfig:
    """Build a minimal, valid ExperimentConfig for runner tests."""
    return ExperimentConfig(
        seed=42,
        dataset=DatasetConfig(name="test_fixture"),
        attacks=attacks if attacks is not None else [],
        retriever=RetrieverConfig(
            type=retriever_type, top_k=top_k, collection_name=collection_name
        ),
        llm=LLMConfig(temperature=temperature, max_tokens=max_tokens),
        metrics=metrics if metrics is not None else [],
        output_dir=str(tmp_path),
        allow_non_benchmark_attacks=allow_non_benchmark_attacks,
    )


def _make_runner(
    tmp_path: Path,
    embedding_dim: int = 8,
    search_results: list[dict] | None = None,
    llm_answer: str = "the answer",
    **config_kwargs: object,
) -> tuple[ExperimentRunner, MagicMock, MagicMock, MagicMock]:
    """
    Build an ExperimentRunner wired to mock embedder/vector_store/llm.

    Returns:
        (runner, mock_embedder, mock_vector_store, mock_llm)
    """
    mock_embedder = MagicMock(spec=BaseEmbedder)
    mock_embedder.embedding_dim = embedding_dim
    mock_embedder.encode.side_effect = lambda texts: [[0.1] * embedding_dim for _ in texts]

    mock_vector_store = MagicMock(spec=BaseVectorStore)
    mock_vector_store.search.return_value = (
        search_results if search_results is not None else []
    )

    mock_llm = MagicMock(spec=BaseLLM)
    mock_llm.model_name = "mock-llm"
    mock_llm.build_rag_prompt.side_effect = (
        lambda query, context_docs: f"PROMPT[{query}|{len(context_docs)} docs]"
    )
    mock_llm.generate.return_value = llm_answer

    config = _make_config(tmp_path, **config_kwargs)  # type: ignore[arg-type]
    runner = ExperimentRunner(
        config=config, embedder=mock_embedder, vector_store=mock_vector_store, llm=mock_llm
    )
    return runner, mock_embedder, mock_vector_store, mock_llm


def _make_search_hit(doc_id: str = "gt_C1", claim_id: str = "C1", text: str = "ctx") -> dict:
    """Build a raw search hit as returned by BaseVectorStore.search()."""
    return {
        "doc_id": doc_id,
        "score": 0.9,
        "payload": {
            "doc_id": doc_id,
            "claim_id": claim_id,
            "text": text,
            "doc_type": "ground_truth",
        },
    }


def _make_poisoned_search_hit(
    doc_id: str = "poisoned_C1", claim_id: str = "C1", text: str = "poisoned ctx"
) -> dict:
    """
    Build a raw search hit that DenseRetriever._document_from_payload will
    reconstruct as a PoisonedDocument (doc_type="poisoned" AND "attack_name"
    present — see that method's own docstring for why both are required).
    """
    return {
        "doc_id": doc_id,
        "score": 0.8,
        "payload": {
            "doc_id": doc_id,
            "claim_id": claim_id,
            "text": text,
            "doc_type": "poisoned",
            "attack_name": "numerical_shift",
            "attack_params": {},
            "original_text": "clean version",
        },
    }


# ─── Initialisation ───────────────────────────────────────────────────────────

class TestExperimentRunnerInit:
    """Tests for __init__ attribute storage and internal wiring."""

    def test_stores_config(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        assert runner.config.dataset.name == "test_fixture"

    def test_stores_embedder_vector_store_llm(self, tmp_path: Path) -> None:
        runner, mock_embedder, mock_vector_store, mock_llm = _make_runner(tmp_path)
        assert runner.embedder is mock_embedder
        assert runner.vector_store is mock_vector_store
        assert runner.llm is mock_llm

    def test_faithfulness_scorer_defaults_to_none(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        assert runner.faithfulness_scorer is None

    def test_collection_from_config(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path, collection_name="my_collection")
        assert runner.collection == "my_collection"

    def test_top_k_from_config(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path, top_k=9)
        assert runner.top_k == 9

    def test_ingestion_pipeline_shares_embedder_and_store(self, tmp_path: Path) -> None:
        runner, mock_embedder, mock_vector_store, _ = _make_runner(tmp_path)
        assert runner.ingestion_pipeline.embedder is mock_embedder
        assert runner.ingestion_pipeline.vector_store is mock_vector_store
        assert runner.ingestion_pipeline.collection == runner.collection

    def test_retriever_shares_embedder_and_store(self, tmp_path: Path) -> None:
        runner, mock_embedder, mock_vector_store, _ = _make_runner(tmp_path)
        assert runner.retriever.embedder is mock_embedder
        assert runner.retriever.vector_store is mock_vector_store
        assert runner.retriever.collection == runner.collection

    def test_retriever_type_defaults_to_dense(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        assert runner.retriever_type == "dense"

    def test_sparse_retriever_type_builds_sparse_retriever(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path, retriever_type="sparse")
        assert isinstance(runner.retriever, SparseRetriever)
        assert runner.retriever_type == "sparse"

    def test_hybrid_retriever_type_builds_hybrid_retriever(self, tmp_path: Path) -> None:
        runner, mock_embedder, mock_vector_store, _ = _make_runner(
            tmp_path, retriever_type="hybrid"
        )
        assert isinstance(runner.retriever, HybridRetriever)
        assert runner.retriever_type == "hybrid"
        # HybridRetriever composes a real DenseRetriever (sharing the same
        # embedder/vector_store/collection as a pure "dense" run) and a
        # real SparseRetriever, per its __init__.
        assert isinstance(runner.retriever.dense, DenseRetriever)
        assert isinstance(runner.retriever.sparse, SparseRetriever)
        assert runner.retriever.dense.embedder is mock_embedder
        assert runner.retriever.dense.vector_store is mock_vector_store
        assert runner.retriever.dense.collection == runner.collection


# ─── run() — corpus / attack resolution ───────────────────────────────────────

class TestRunCorpusBuilding:
    """Tests for attack resolution and corpus construction in run()."""

    def test_no_attacks_produces_only_ground_truth(self, tmp_path: Path) -> None:
        runner, _, mock_vector_store, _ = _make_runner(tmp_path)
        claims = [_make_claim("C1"), _make_claim("C2")]
        runner.run(claims, save=False)
        # upsert receives all_documents; with no attacks, that's one
        # ground-truth doc per claim.
        args, _ = mock_vector_store.upsert.call_args
        assert len(args[1]) == 2

    def test_registered_attack_produces_poisoned_documents(self, tmp_path: Path) -> None:
        runner, _, mock_vector_store, _ = _make_runner(
            tmp_path,
            attacks=[AttackConfig(name="numerical_shift", poison_rate=1.0)],
        )
        claims = [_make_claim("C1", text="Inflation rose to 3.5% in 2023.")]
        runner.run(claims, save=False)
        args, _ = mock_vector_store.upsert.call_args
        # 1 ground-truth + 1 poisoned (poison_rate=1.0 always poisons).
        assert len(args[1]) == 2

    def test_unregistered_attack_raises(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(
            tmp_path,
            attacks=[AttackConfig(name="does_not_exist", poison_rate=1.0)],
        )
        with pytest.raises(AttackNotFoundError):
            runner.run([_make_claim()], save=False)

    def test_non_benchmark_attack_raises_configuration_error_by_default(
        self, tmp_path: Path
    ) -> None:
        """
        attribution_switch (M03) is excluded_from_benchmark — run() must
        refuse it unless config.allow_non_benchmark_attacks is True (see
        eiger/ingestion/corpus_builder.py).
        """
        from eiger.core.exceptions import ConfigurationError

        runner, _, _, _ = _make_runner(
            tmp_path,
            attacks=[AttackConfig(name="attribution_switch", poison_rate=1.0)],
        )
        with pytest.raises(ConfigurationError, match="attribution_switch"):
            runner.run([_make_claim()], save=False)

    def test_non_benchmark_attack_allowed_with_explicit_opt_in(
        self, tmp_path: Path
    ) -> None:
        runner, _, mock_vector_store, _ = _make_runner(
            tmp_path,
            attacks=[AttackConfig(name="attribution_switch", poison_rate=1.0)],
            allow_non_benchmark_attacks=True,
        )
        # The claim must mention a mapped entity, otherwise attribution_switch
        # is a no-op and the poisoned copy is skipped by CorpusBuilder.
        runner.run([_make_claim(text="The WHO reported inflation rose to 3.5% in 2023.")], save=False)
        args, _ = mock_vector_store.upsert.call_args
        # 1 ground-truth + 1 poisoned.
        assert len(args[1]) == 2


# ─── run() — ingestion ─────────────────────────────────────────────────────────

class TestRunIngestion:
    """Tests for the ingestion step of run()."""

    def test_resets_collection(self, tmp_path: Path) -> None:
        runner, _, mock_vector_store, _ = _make_runner(tmp_path, embedding_dim=16)
        runner.run([_make_claim()], save=False)
        args, kwargs = mock_vector_store.reset_collection.call_args
        assert args[0] == runner.collection
        assert kwargs["dim"] == 16

    def test_upserts_documents(self, tmp_path: Path) -> None:
        runner, _, mock_vector_store, _ = _make_runner(tmp_path)
        runner.run([_make_claim()], save=False)
        mock_vector_store.upsert.assert_called_once()


# ─── run() — sparse retriever path ─────────────────────────────────────────────

class TestRunSparseRetriever:
    """Tests for run() when config.retriever.type == 'sparse' (Sprint 4)."""

    def test_sparse_run_skips_vector_store_ingestion(self, tmp_path: Path) -> None:
        """
        A sparse-only run must not touch the vector store at all — that is
        the entire point of SparseRetriever owning its own in-memory index
        (see its module docstring).
        """
        runner, _, mock_vector_store, _ = _make_runner(tmp_path, retriever_type="sparse")
        runner.run([_make_claim()], save=False)
        mock_vector_store.reset_collection.assert_not_called()
        mock_vector_store.upsert.assert_not_called()

    def test_sparse_run_ranks_by_lexical_overlap(self, tmp_path: Path) -> None:
        """
        End-to-end proof that ExperimentRunner actually drives BM25 ranking:
        two claims share the same context_query, but only one's ground-truth
        document shares vocabulary with it. SparseRetriever must retrieve
        that document (and not the other) for both claims.
        """
        runner, _, _, mock_llm = _make_runner(tmp_path, retriever_type="sparse", top_k=1)
        claim_a = Claim(
            claim_id="A",
            original_fact="The central bank raised interest rates to control inflation.",
            context_query="What happened to inflation?",
            source_dataset="test_fixture",
        )
        claim_b = Claim(
            claim_id="B",
            original_fact="The football team won the championship match yesterday.",
            context_query="What happened to inflation?",
            source_dataset="test_fixture",
        )
        runner.run([claim_a, claim_b], save=False)
        for call in mock_llm.build_rag_prompt.call_args_list:
            _, context_docs = call.args
            assert context_docs == [
                "The central bank raised interest rates to control inflation."
            ]

    def test_sparse_run_with_no_claims_produces_empty_result(self, tmp_path: Path) -> None:
        runner, _, _, mock_llm = _make_runner(tmp_path, retriever_type="sparse")
        result = runner.run([], save=False)
        assert result.records == []
        mock_llm.build_rag_prompt.assert_not_called()


# ─── run() — hybrid retriever path ─────────────────────────────────────────────

class TestRunHybridRetriever:
    """Tests for run() when config.retriever.type == 'hybrid' (Sprint 5)."""

    def test_hybrid_run_ingests_vector_store_and_fits_sparse_index(self, tmp_path: Path) -> None:
        """
        A hybrid run must populate BOTH indexes: the vector store (dense
        side, like a pure "dense" run) and the in-memory BM25 index (sparse
        side, like a pure "sparse" run) — unlike a sparse-only run, which
        skips the vector store entirely (see TestRunSparseRetriever above).
        """
        runner, _, mock_vector_store, _ = _make_runner(tmp_path, retriever_type="hybrid")
        runner.run([_make_claim()], save=False)
        mock_vector_store.reset_collection.assert_called_once()
        mock_vector_store.upsert.assert_called_once()

    def test_hybrid_run_ranks_by_lexical_overlap_when_dense_returns_nothing(
        self, tmp_path: Path
    ) -> None:
        """
        End-to-end proof that HybridRetriever's sparse side is actually
        wired in: with the mocked vector store returning zero dense hits
        (the default when `search_results` is not overridden), the fused
        ranking must still surface the lexically-relevant document via BM25
        — mirroring TestRunSparseRetriever.test_sparse_run_ranks_by_lexical_overlap.
        """
        runner, _, _, mock_llm = _make_runner(tmp_path, retriever_type="hybrid", top_k=1)
        claim_a = Claim(
            claim_id="A",
            original_fact="The central bank raised interest rates to control inflation.",
            context_query="What happened to inflation?",
            source_dataset="test_fixture",
        )
        claim_b = Claim(
            claim_id="B",
            original_fact="The football team won the championship match yesterday.",
            context_query="What happened to inflation?",
            source_dataset="test_fixture",
        )
        runner.run([claim_a, claim_b], save=False)
        for call in mock_llm.build_rag_prompt.call_args_list:
            _, context_docs = call.args
            assert context_docs == [
                "The central bank raised interest rates to control inflation."
            ]

    def test_hybrid_run_with_no_claims_produces_empty_result(self, tmp_path: Path) -> None:
        runner, _, _, mock_llm = _make_runner(tmp_path, retriever_type="hybrid")
        result = runner.run([], save=False)
        assert result.records == []
        mock_llm.build_rag_prompt.assert_not_called()


# ─── run() — retrieval & generation ────────────────────────────────────────────

class TestRunRetrievalAndGeneration:
    """Tests for the per-claim retrieval/generation step of run()."""

    def test_llm_generate_receives_configured_temperature_and_max_tokens(
        self, tmp_path: Path
    ) -> None:
        runner, _, _, mock_llm = _make_runner(tmp_path, temperature=0.42, max_tokens=99)
        runner.run([_make_claim()], save=False)
        _, kwargs = mock_llm.generate.call_args
        assert kwargs["temperature"] == 0.42
        assert kwargs["max_tokens"] == 99

    def test_build_rag_prompt_called_with_query_and_context(self, tmp_path: Path) -> None:
        hit = _make_search_hit(text="retrieved context")
        runner, _, _, mock_llm = _make_runner(tmp_path, search_results=[hit])
        claim = _make_claim("C1")
        runner.run([claim], save=False)
        args, _ = mock_llm.build_rag_prompt.call_args
        assert args[0] == claim.context_query
        assert args[1] == ["retrieved context"]

    def test_generation_result_fields(self, tmp_path: Path) -> None:
        hit = _make_search_hit(text="ctx text")
        runner, _, _, mock_llm = _make_runner(
            tmp_path, search_results=[hit], llm_answer="42% inflation"
        )
        claim = _make_claim("C1")
        result = runner.run([claim], save=False)
        record = result.records[0]
        assert record.generation.claim_id == "C1"
        assert record.generation.query == claim.context_query
        assert record.generation.context_docs == ["ctx text"]
        assert record.generation.answer == "42% inflation"
        assert record.generation.model_name == "mock-llm"
        assert record.generation.metadata["temperature"] == runner.config.llm.temperature
        assert record.generation.metadata["max_tokens"] == runner.config.llm.max_tokens
        # Source Integrity reads the claim text from here (review fix).
        assert record.generation.metadata["ground_truth"] == claim.original_fact

    def test_one_record_per_claim(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        claims = [_make_claim("C1"), _make_claim("C2"), _make_claim("C3")]
        result = runner.run(claims, save=False)
        assert len(result.records) == 3
        assert [r.claim_id for r in result.records] == ["C1", "C2", "C3"]


# ─── run() — metrics ────────────────────────────────────────────────────────────

class TestRunMetrics:
    """Tests for metric resolution and computation in run()."""

    def test_unregistered_metric_raises(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path, metrics=["does_not_exist"])
        with pytest.raises(MetricNotFoundError):
            runner.run([_make_claim()], save=False)

    def test_ffr_written_back_into_record_metrics(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path, metrics=["ffr"])
        result = runner.run([_make_claim()], save=False)
        assert "ffr" in result.records[0].metrics
        assert "ffr" in result.aggregate_metrics

    def test_ffr_defaults_to_zero_without_scorer(self, tmp_path: Path) -> None:
        """
        Without a faithfulness_scorer, faithfulness/correctness default to
        0.0, so FFR (which requires faithfulness > 0.8) must be 0.0.
        """
        runner, _, _, _ = _make_runner(tmp_path, metrics=["ffr"])
        result = runner.run([_make_claim()], save=False)
        assert result.aggregate_metrics["ffr"] == 0.0

    def test_faithfulness_scorer_output_drives_ffr(self, tmp_path: Path) -> None:
        def scorer(claim: Claim, generation: GenerationResult) -> dict[str, float]:
            return {"ragas_faithfulness": 0.95, "ragas_answer_correctness": 0.05}

        runner, _, _, _ = _make_runner(tmp_path, metrics=["ffr"])
        runner.faithfulness_scorer = scorer
        result = runner.run([_make_claim()], save=False)
        # faithfulness 0.95 > 0.8 and correctness 0.05 < 0.2 => faithful falsehood.
        assert result.aggregate_metrics["ffr"] == 1.0
        assert result.records[0].metrics["ragas_faithfulness"] == 0.95

    def test_warns_when_ffr_configured_without_scorer(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path, metrics=["ffr"])
        with patch("eiger.experiments.runner.log") as mock_log:
            runner.run([_make_claim()], save=False)
        mock_log.warning.assert_called_once()

    def test_no_warning_when_scorer_provided(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path, metrics=["ffr"])
        runner.faithfulness_scorer = lambda claim, generation: {}
        with patch("eiger.experiments.runner.log") as mock_log:
            runner.run([_make_claim()], save=False)
        mock_log.warning.assert_not_called()

    def test_ers_runs_without_annotations_and_returns_zero(self, tmp_path: Path) -> None:
        """ERS on a corpus with no PoisonAnnotations must gracefully return 0.0."""
        runner, _, _, _ = _make_runner(tmp_path, metrics=["ers"])
        result = runner.run([_make_claim()], save=False)
        assert result.aggregate_metrics["ers"] == 0.0


# ─── run() — PCS counterfactual generation ─────────────────────────────────────

class TestRunPCS:
    """
    Tests for _add_counterfactual_generation / _resolve_metric — the PCS
    (Poisoned Context Sensitivity) support added alongside eiger/metrics/pcs.py.
    """

    def test_no_second_generation_when_pcs_not_configured(self, tmp_path: Path) -> None:
        """Even with a poisoned hit retrieved, "pcs" must be opt-in."""
        hit = _make_poisoned_search_hit()
        runner, _, _, mock_llm = _make_runner(tmp_path, search_results=[hit], metrics=[])
        runner.run([_make_claim()], save=False)
        assert mock_llm.generate.call_count == 1

    def test_no_second_generation_when_no_poisoned_hit(self, tmp_path: Path) -> None:
        """With "pcs" configured but nothing poisoned retrieved, skip the extra call."""
        hit = _make_search_hit()  # ground_truth only
        runner, _, _, mock_llm = _make_runner(tmp_path, search_results=[hit], metrics=["pcs"])
        runner.run([_make_claim()], save=False)
        assert mock_llm.generate.call_count == 1

    def test_second_generation_when_pcs_configured_and_poisoned_hit_present(
        self, tmp_path: Path
    ) -> None:
        hit = _make_poisoned_search_hit()
        runner, _, _, mock_llm = _make_runner(tmp_path, search_results=[hit], metrics=["pcs"])
        runner.run([_make_claim()], save=False)
        assert mock_llm.generate.call_count == 2

    def test_counterfactual_prompt_excludes_poisoned_docs(self, tmp_path: Path) -> None:
        clean_hit = _make_search_hit(doc_id="gt", text="clean text")
        poisoned_hit = _make_poisoned_search_hit(doc_id="poison", text="poisoned text")
        runner, _, _, mock_llm = _make_runner(
            tmp_path, search_results=[clean_hit, poisoned_hit], metrics=["pcs"]
        )
        runner.run([_make_claim()], save=False)

        # First call: real generation, both docs. Second: counterfactual,
        # poisoned doc filtered out.
        first_args, _ = mock_llm.build_rag_prompt.call_args_list[0]
        second_args, _ = mock_llm.build_rag_prompt.call_args_list[1]
        assert first_args[1] == ["clean text", "poisoned text"]
        assert second_args[1] == ["clean text"]

    def test_counterfactual_answer_stashed_on_generation_metadata(self, tmp_path: Path) -> None:
        hit = _make_poisoned_search_hit()
        runner, _, _, mock_llm = _make_runner(tmp_path, search_results=[hit], metrics=["pcs"])
        mock_llm.generate.side_effect = ["real answer", "counterfactual answer"]
        result = runner.run([_make_claim()], save=False)
        generation = result.records[0].generation
        assert generation.answer == "real answer"
        assert generation.metadata["counterfactual_answer"] == "counterfactual answer"
        assert generation.metadata["counterfactual_context_docs"] == []

    def test_counterfactual_generation_uses_configured_temperature_and_max_tokens(
        self, tmp_path: Path
    ) -> None:
        hit = _make_poisoned_search_hit()
        runner, _, _, mock_llm = _make_runner(
            tmp_path, search_results=[hit], metrics=["pcs"], temperature=0.3, max_tokens=77
        )
        runner.run([_make_claim()], save=False)
        for call in mock_llm.generate.call_args_list:
            assert call.kwargs["temperature"] == 0.3
            assert call.kwargs["max_tokens"] == 77

    def test_pcs_aggregate_metric_computed_end_to_end(self, tmp_path: Path) -> None:
        hit = _make_poisoned_search_hit()
        runner, _, _, mock_llm = _make_runner(tmp_path, search_results=[hit], metrics=["pcs"])
        mock_llm.generate.side_effect = ["real answer", "totally different"]
        result = runner.run([_make_claim()], save=False)
        assert "pcs" in result.aggregate_metrics
        assert "pcs" in result.records[0].metrics
        # Real embedder is mocked to return the identical vector for every
        # text (see _make_runner's mock_embedder.encode), so cosine
        # similarity is ~1.0 regardless of text content -> PCS ~0.0.
        # pytest.approx (not exact equality) because summing many identical
        # floating-point terms in raw_cosine's dot/norm computation does not
        # exactly cancel to 1.0 -- a few ULPs of error survive the (raw + 1)
        # / 2 rescale and the 1 - similarity subtraction. This test only
        # asserts the metric ran end-to-end, not a specific non-trivial
        # value (that is covered precisely, with hand-picked vectors, in
        # test_pcs.py).
        assert result.aggregate_metrics["pcs"] == pytest.approx(0.0, abs=1e-9)

    def test_resolve_metric_pcs_bypasses_registry(self, tmp_path: Path) -> None:
        from eiger.metrics.pcs import PCSMetric

        runner, mock_embedder, _, _ = _make_runner(tmp_path)
        metric = runner._resolve_metric("pcs")
        assert isinstance(metric, PCSMetric)
        assert metric.embedder is mock_embedder

    def test_resolve_metric_other_names_use_registry(self, tmp_path: Path) -> None:
        from eiger.metrics import FFRMetric

        runner, _, _, _ = _make_runner(tmp_path)
        metric = runner._resolve_metric("ffr")
        assert isinstance(metric, FFRMetric)

    def test_resolve_metric_unknown_name_raises(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        with pytest.raises(MetricNotFoundError):
            runner._resolve_metric("does_not_exist")


# ─── run() — result assembly & empty input ────────────────────────────────────

class TestRunResultAssembly:
    """Tests for the ExperimentResult object returned by run()."""

    def test_returns_experiment_result(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        result = runner.run([_make_claim()], save=False)
        assert isinstance(result, ExperimentResult)

    def test_experiment_id_and_config_hash_match_config(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        result = runner.run([_make_claim()], save=False)
        assert result.experiment_id == runner.config.experiment_id
        assert result.config_hash == runner.config.config_hash

    def test_environment_captured(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        result = runner.run([_make_claim()], save=False)
        assert "python_version" in result.environment
        assert "platform" in result.environment

    def test_empty_claims_produces_empty_but_valid_result(self, tmp_path: Path) -> None:
        runner, _, mock_vector_store, mock_llm = _make_runner(tmp_path)
        result = runner.run([], save=False)
        assert result.records == []
        mock_llm.generate.assert_not_called()
        # reset_collection still happens (clean corpus per run), upsert does not
        # (IngestionPipeline skips upsert entirely for an empty document list).
        mock_vector_store.reset_collection.assert_called_once()
        mock_vector_store.upsert.assert_not_called()


# ─── _get_git_commit ───────────────────────────────────────────────────────────

class TestGetGitCommit:
    """Tests for the git commit SHA capture helper."""

    def test_returns_stripped_stdout_on_success(self) -> None:
        fake_completed = MagicMock()
        fake_completed.stdout = "abc123\n"
        with patch("eiger.experiments.runner.subprocess.run", return_value=fake_completed):
            assert ExperimentRunner._get_git_commit() == "abc123"

    def test_returns_unknown_on_failure(self) -> None:
        with patch(
            "eiger.experiments.runner.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "git"),
        ):
            assert ExperimentRunner._get_git_commit() == "unknown"

    def test_returns_unknown_when_git_missing(self) -> None:
        with patch(
            "eiger.experiments.runner.subprocess.run", side_effect=FileNotFoundError()
        ):
            assert ExperimentRunner._get_git_commit() == "unknown"


# ─── _capture_environment ──────────────────────────────────────────────────────

class TestCaptureEnvironment:
    """Tests for the environment metadata capture helper."""

    def test_contains_expected_keys(self) -> None:
        env = ExperimentRunner._capture_environment()
        assert "python_version" in env
        assert "platform" in env
        assert isinstance(env["python_version"], str)
        assert isinstance(env["platform"], str)


# ─── save_result / run(save=...) ───────────────────────────────────────────────

class TestSaveResult:
    """Tests for result persistence."""

    def test_run_save_true_writes_results_json(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        runner.run([_make_claim()], save=True)
        result_path = tmp_path / "results.json"
        assert result_path.exists()
        data = json.loads(result_path.read_text())
        assert data["experiment_id"] == runner.config.experiment_id

    def test_run_save_false_does_not_write(self, tmp_path: Path) -> None:
        runner, _, _, _ = _make_runner(tmp_path)
        runner.run([_make_claim()], save=False)
        assert not (tmp_path / "results.json").exists()

    def test_save_result_creates_output_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "nested" / "dir"
        runner, _, _, _ = _make_runner(tmp_path)
        runner.config.output_dir = str(nested)
        result = runner.run([_make_claim()], save=False)
        path = runner.save_result(result)
        assert path == nested / "results.json"
        assert path.exists()
