"""
Unit tests for SourceIntegrityMetric (eiger.metrics.source_integrity).

Tests verify:
  - Default model name and lazy-init state (pipeline starts as None)
  - Custom model name is stored correctly
  - _load_pipeline is idempotent: calling it twice does not overwrite a
    pipeline that was already loaded
  - _load_pipeline emits a UserWarning and leaves _pipeline=None when the
    ``transformers`` package is not installed (simulated via sys.modules)
  - compute() returns 0.0 with warning metadata when ground truth is absent
    (empty retrieval query)
  - compute() returns 0.0 with warning metadata when the NLI pipeline is
    unavailable (simulated by leaving _pipeline=None)
  - compute() returns 0.0 for records with no retrieval hits
  - compute() returns the correct mean consistency score when the pipeline
    is mocked to return fixed scores
  - _get_ground_truth returns None for an empty query and the query string
    for a non-empty query
  - aggregate() returns 0.0 for an empty list and the mean for a non-empty list

What these tests do NOT cover:
  - Real NLI model inference (requires ~90 MB download and GPU/CPU time;
    covered in integration tests).
  - CUDA device placement or multi-GPU setups.
"""

from __future__ import annotations

import sys
import warnings
from unittest.mock import MagicMock

import pytest

from eiger.core.models import (
    Document,
    EvaluationRecord,
    GenerationResult,
    MetricScore,
    RetrievalResult,
    RetrievedDocument,
)
from eiger.metrics.source_integrity import SourceIntegrityMetric


# ─── Test helpers ─────────────────────────────────────────────────────────────

_GT = "The WHO reported that inflation rose to 3.5% in 2023."


def _nli(entail: float) -> list[dict]:
    """Shape returned by the text-classification pipeline with top_k=None."""
    rest = (1.0 - entail) / 2
    return [
        {"label": "contradiction", "score": rest},
        {"label": "entailment", "score": entail},
        {"label": "neutral", "score": rest},
    ]


def _make_record(
    query: str = "What is the inflation rate?",
    with_hits: bool = True,
    ground_truth: str | None = _GT,
) -> EvaluationRecord:
    """
    Build a minimal EvaluationRecord for SourceIntegrityMetric tests.

    Args:
        query:     The retrieval query string. Pass an empty string to simulate
                   an absent ground truth (SourceIntegrityMetric._get_ground_truth
                   returns None when the query is empty/falsy).
        with_hits: If True, the retrieval result includes one document hit.
                   If False, the hits list is empty (simulates a failed retrieval).

    Returns:
        A minimal EvaluationRecord with no pre-computed metrics.
    """
    doc = Document(
        claim_id="C1",
        text="The WHO reported that inflation rose to 3.5% in 2023.",
        doc_type="ground_truth",
    )
    hits = [RetrievedDocument(document=doc, score=0.9, rank=1)] if with_hits else []
    retrieval = RetrievalResult(query=query, claim_id="C1", hits=hits, top_k=1)
    generation = GenerationResult(
        claim_id="C1",
        query=query,
        context_docs=[doc.text],
        answer="Inflation rose to 3.5%.",
        model_name="test_model",
        metadata={} if ground_truth is None else {"ground_truth": ground_truth},
    )
    return EvaluationRecord(
        claim_id="C1",
        generation=generation,
        retrieval=retrieval,
        metrics={},
    )


# ─── Initialisation ───────────────────────────────────────────────────────────

class TestSourceIntegrityMetricInit:
    """Tests for __init__ and attribute defaults."""

    def test_default_model_name(self) -> None:
        """The default NLI model must match the module-level constant."""
        metric = SourceIntegrityMetric()
        assert metric.model_name == "cross-encoder/nli-MiniLM2-L6-H768"

    def test_pipeline_starts_as_none(self) -> None:
        """
        _pipeline must be None at construction time (lazy-loading contract).

        Eager loading would trigger a ~90 MB download on every import,
        which is unacceptable for CI environments.
        """
        metric = SourceIntegrityMetric()
        assert metric._pipeline is None

    def test_custom_model_name_stored(self) -> None:
        """A custom model name passed to __init__ must be stored on the instance."""
        metric = SourceIntegrityMetric(model_name="my-org/custom-nli-model")
        assert metric.model_name == "my-org/custom-nli-model"


# ─── _load_pipeline ───────────────────────────────────────────────────────────

class TestLoadPipeline:
    """Tests for the lazy NLI pipeline loader."""

    def test_idempotent_when_pipeline_already_loaded(self) -> None:
        """
        Calling _load_pipeline when self._pipeline is already set must return
        immediately without overwriting the existing pipeline.

        This guards against accidentally re-downloading the model in long-running
        experiment loops that call compute() many times.
        """
        metric = SourceIntegrityMetric()
        mock_pipe = MagicMock()
        metric._pipeline = mock_pipe
        metric._load_pipeline()  # must be a no-op
        # Pipeline reference must be unchanged
        assert metric._pipeline is mock_pipe

    def test_warns_and_leaves_none_when_transformers_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        If ``transformers`` is not importable (simulated via sys.modules),
        _load_pipeline must emit a UserWarning and leave _pipeline as None.

        The UserWarning is the correct signal level here: it is non-fatal,
        allowing experiments that mix SI with other metrics to still run
        partially when transformers is absent.
        """
        # Setting sys.modules['transformers'] = None causes any subsequent
        # ``import transformers`` or ``from transformers import ...`` to raise
        # ImportError, which is the standard Python way to simulate a missing package.
        monkeypatch.setitem(sys.modules, "transformers", None)
        metric = SourceIntegrityMetric()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metric._load_pipeline()
        assert metric._pipeline is None
        # At least one warning must mention transformers
        assert any("transformers" in str(w.message).lower() for w in caught)

    def test_load_pipeline_success_when_transformers_available(self) -> None:
        """_load_pipeline must set _pipeline when transformers loads successfully.

        ``pipeline`` is imported *inside* the method body
        (``from transformers import pipeline as hf_pipeline``), so we cannot
        patch a module-level attribute.  Instead we replace the entire
        ``transformers`` entry in ``sys.modules`` with a MagicMock whose
        ``.pipeline`` attribute returns our sentinel instance.
        """
        import sys
        from unittest.mock import patch, MagicMock

        mock_pipe_instance = MagicMock()
        mock_hf_pipeline = MagicMock(return_value=mock_pipe_instance)
        mock_transformers = MagicMock()
        mock_transformers.pipeline = mock_hf_pipeline

        # Replace transformers in sys.modules so the local import inside
        # _load_pipeline picks up our mock.  Also silence the module-level
        # ``log`` to avoid PrintLogger.name errors from structlog processors.
        with patch.dict(sys.modules, {"transformers": mock_transformers}), \
             patch("eiger.metrics.source_integrity.log"):
            metric = SourceIntegrityMetric()
            metric._load_pipeline()   # exercises source_integrity.py:122-128
        assert metric._pipeline is mock_pipe_instance

# ─── compute ─────────────────────────────────────────────────────────────────

class TestCompute:
    """Tests for the SourceIntegrityMetric.compute() method."""

    def test_no_ground_truth_returns_zero_with_warning(self) -> None:
        """
        compute() must return 0.0 with warning metadata when the retrieval query
        is empty (which makes _get_ground_truth return None).

        Without a ground-truth string, NLI comparison is impossible, so the
        safe default is 0.0 with an explanatory metadata entry.
        """
        metric = SourceIntegrityMetric()
        # Inject a mock pipeline so that the import check does not fail
        metric._pipeline = MagicMock()
        record = _make_record(ground_truth=None)  # no stored ground truth
        score = metric.compute(record)
        assert score.value == 0.0
        assert "warning" in score.metadata

    def test_pipeline_none_returns_zero_with_warning(self) -> None:
        """
        compute() must return 0.0 with warning metadata when _pipeline is None
        (i.e. _load_pipeline could not load the model).

        This exercises the ``if self._pipeline is None`` branch that fires after
        the ImportError fallback in _load_pipeline.
        """
        metric = SourceIntegrityMetric()
        # Prevent _load_pipeline from attempting to download the model
        metric._load_pipeline = lambda: None  # type: ignore[method-assign]
        record = _make_record(query="What is the inflation rate?")
        # _pipeline is still None; compute must handle this gracefully
        score = metric.compute(record)
        assert score.value == 0.0
        assert "warning" in score.metadata

    def test_empty_hits_returns_zero(self) -> None:
        """
        compute() must return 0.0 when the retrieval result has no hits.

        An empty hit list means there are no documents to score; returning
        0.0 is the conservative choice (no evidence of integrity).
        """
        metric = SourceIntegrityMetric()
        metric._pipeline = MagicMock()
        record = _make_record(query="Some query", with_hits=False)
        score = metric.compute(record)
        assert score.value == 0.0

    def test_compute_with_mock_pipeline_returns_correct_mean(self) -> None:
        """
        compute() must return the correct mean consistency score when the
        pipeline returns a fixed label/score dict.

        The mock pipeline always returns consistency=0.8, so the SI for a
        single-document retrieval must be exactly 0.8.
        """
        metric = SourceIntegrityMetric()
        # Mock the pipeline to always return 0.8 consistency score
        mock_pipe = MagicMock(return_value=_nli(0.8))
        metric._pipeline = mock_pipe
        record = _make_record()
        score = metric.compute(record)
        assert score.value == pytest.approx(0.8)
        # Premise is the retrieved document, hypothesis is the ground truth
        # (NOT the query), and all labels are requested.
        mock_pipe.assert_called_once_with(
            {"text": record.retrieval.hits[0].document.text, "text_pair": _GT}, top_k=None
        )

    def test_entailment_found_by_label_name_and_nested_output(self) -> None:
        """Label order must not matter, and a nested [[...]] output is unwrapped."""
        metric = SourceIntegrityMetric()
        metric._pipeline = MagicMock(return_value=[list(reversed(_nli(0.3)))])
        assert metric.compute(_make_record()).value == pytest.approx(0.3)

    def test_model_without_entailment_label_raises(self) -> None:
        """A non-NLI model must fail loudly instead of reporting a wrong score."""
        metric = SourceIntegrityMetric()
        metric._pipeline = MagicMock(return_value=[{"label": "POSITIVE", "score": 0.9}])
        with pytest.raises(ValueError, match="no entailment label"):
            metric.compute(_make_record())

    def test_compute_metadata_contains_entailment_scores(self) -> None:
        """
        compute() must include per-document entailment_scores and n_documents
        in the MetricScore metadata for post-hoc analysis.
        """
        metric = SourceIntegrityMetric()
        metric._pipeline = MagicMock(return_value=_nli(0.7))
        record = _make_record(query="Some query")
        score = metric.compute(record)
        assert "entailment_scores" in score.metadata
        assert "n_documents" in score.metadata
        assert score.metadata["n_documents"] == 1

    def test_compute_metric_name_is_source_integrity(self) -> None:
        """The metric_name field on the returned MetricScore must be 'source_integrity'."""
        metric = SourceIntegrityMetric()
        metric._pipeline = MagicMock()
        record = _make_record(query="Some query", with_hits=False)
        score = metric.compute(record)
        assert score.metric_name == "source_integrity"


# ─── _get_ground_truth ────────────────────────────────────────────────────────

class TestGetGroundTruth:
    """Tests for the static _get_ground_truth helper."""

    def test_returns_none_without_stored_ground_truth(self) -> None:
        """No stored ground truth -> None, even when a query exists."""
        record = _make_record(query="What is the inflation rate?", ground_truth=None)
        assert SourceIntegrityMetric._get_ground_truth(record) is None

    def test_returns_none_for_empty_ground_truth(self) -> None:
        assert SourceIntegrityMetric._get_ground_truth(_make_record(ground_truth="")) is None

    def test_returns_stored_ground_truth_not_query(self) -> None:
        """The claim text must be used, never the retrieval query."""
        record = _make_record(query="What is the inflation rate?")
        assert SourceIntegrityMetric._get_ground_truth(record) == _GT


# ─── aggregate ───────────────────────────────────────────────────────────────

class TestAggregate:
    """Tests for the SourceIntegrityMetric.aggregate() method."""

    def test_empty_list_returns_zero(self) -> None:
        """aggregate() must return 0.0 for an empty list (safe default, no ZeroDivisionError)."""
        metric = SourceIntegrityMetric()
        assert metric.aggregate([]) == 0.0

    def test_single_score_returns_that_value(self) -> None:
        """aggregate() of a single-element list must return the element's value."""
        metric = SourceIntegrityMetric()
        scores = [MetricScore(metric_name="source_integrity", value=0.75)]
        assert metric.aggregate(scores) == pytest.approx(0.75)

    def test_multiple_scores_returns_mean(self) -> None:
        """aggregate() must return the arithmetic mean of all score values."""
        metric = SourceIntegrityMetric()
        scores = [
            MetricScore(metric_name="source_integrity", value=0.8),
            MetricScore(metric_name="source_integrity", value=0.6),
        ]
        assert metric.aggregate(scores) == pytest.approx(0.7)
