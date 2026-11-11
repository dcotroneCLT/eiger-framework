"""Regression tests for the November 2026 code review fixes."""

from __future__ import annotations

from eiger.attacks import get_attack
from eiger.core.models import AttackConfig, Claim, RetrievalResult
from eiger.ingestion.corpus_builder import CorpusBuilder
from eiger.retrieval.hybrid_retriever import HybridRetriever
from eiger.retrieval.sparse_retriever import SparseRetriever

_CLAIM = Claim(
    claim_id="C1",
    original_fact="Inflation rose to 3.5% in 2023 according to the 2023 report.",
    context_query="inflation 2023 report",
)


def _corpus():
    attacks = [
        (get_attack(name), AttackConfig(name=name, poison_rate=1.0))
        for name in ("numerical_shift", "date_manipulation")
    ]
    return CorpusBuilder(attacks).build([_CLAIM])


def test_poisoned_documents_have_unique_ids() -> None:
    ids = [d.doc_id for d in _corpus().all_documents]
    assert ids == ["gt_C1", "gt_C1__numerical_shift", "gt_C1__date_manipulation"]
    assert len(set(ids)) == len(ids)


class _EmptyDense:
    def retrieve(self, query: str, claim_id: str, top_k: int) -> RetrievalResult:
        return RetrievalResult(query=query, claim_id=claim_id, hits=[], top_k=top_k)


def test_hybrid_no_longer_hides_poisoned_documents() -> None:
    """Before the fix RRF fusion keyed by doc_id merged poison into ground truth."""
    corpus = _corpus()
    hybrid = HybridRetriever(dense=_EmptyDense(), sparse=SparseRetriever())  # type: ignore[arg-type]
    hybrid.fit(corpus.all_documents)
    result = hybrid.retrieve("inflation 2023 report", "C1", top_k=5)
    assert len(result.hits) == 3
    assert result.contains_poisoned


def test_noop_attacks_are_skipped_and_counted() -> None:
    """An attack that changes nothing must not produce a 'poisoned' copy of the truth."""
    claim = Claim(claim_id="N1", original_fact="The museum reopened after renovation.",
                  context_query="museum reopening")
    attacks = [(get_attack("numerical_shift"), AttackConfig(name="numerical_shift", poison_rate=1.0))]
    result = CorpusBuilder(attacks).build([claim])
    assert result.poisoned_docs == []
    assert result.skipped_noop == {"numerical_shift": 1}
