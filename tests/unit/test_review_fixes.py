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


def test_bm25_ties_are_not_resolved_by_index_order() -> None:
    """
    Two documents that differ only in a token the query does not contain score
    exactly the same. The winner must not depend on which one was indexed first
    (before the fix the ground truth, indexed first, always won).
    """
    from eiger.core.models import Document

    def corpus(truth_first: bool) -> list[Document]:
        gt = Document(doc_id="gt_C1", claim_id="C1", doc_type="ground_truth",
                      text="The central bank said inflation rose to 3.5% in 2023.")
        po = Document(doc_id="gt_C1__x", claim_id="C1", doc_type="poisoned",
                      text="The central bank said inflation rose to 5.3% in 2023.")
        others = [Document(doc_id=f"o{i}", claim_id=f"O{i}", doc_type="ground_truth", text=x)
                  for i, x in enumerate(["Unemployment fell last quarter.",
                                         "The museum reopened after renovation.",
                                         "A vaccine trial started in Europe."])]
        return ([gt, po] if truth_first else [po, gt]) + others

    def first(docs: list[Document]) -> str:
        retriever = SparseRetriever()
        retriever.fit(docs)
        hits = retriever.retrieve("What did the central bank say about inflation in 2023?", "C1", 5).hits
        assert hits[0].score == hits[1].score == 1.0  # a genuine tie
        return hits[0].document.doc_type

    # Same winner whatever the insertion order.
    assert first(corpus(True)) == first(corpus(False))


def test_bm25_tie_break_is_unbiased_across_pairs() -> None:
    """Over many tied pairs, neither the truth nor the poison wins systematically."""
    from eiger.core.models import Document

    truth_wins = 0
    n = 200
    for i in range(n):
        gt = Document(doc_id=f"gt_{i}", claim_id=f"C{i}", doc_type="ground_truth",
                      text=f"Officials confirmed the figure {i}1 for the regional programme.")
        po = Document(doc_id=f"gt_{i}__numerical_shift", claim_id=f"C{i}", doc_type="poisoned",
                      text=f"Officials confirmed the figure {i}9 for the regional programme.")
        filler = [Document(doc_id=f"f{i}_{j}", claim_id=f"F{j}", doc_type="ground_truth", text=x)
                  for j, x in enumerate(["Rain fell on the coast.", "A bridge was repaired.", "Schools reopened."])]
        retriever = SparseRetriever()
        retriever.fit([gt, po] + filler)
        top = retriever.retrieve("What did officials confirm about the regional programme?", f"C{i}", 2).hits[0]
        truth_wins += top.document.doc_type == "ground_truth"
    assert 0.35 < truth_wins / n < 0.65
