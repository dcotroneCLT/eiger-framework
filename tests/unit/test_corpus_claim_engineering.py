"""Tests for the engineering corpus loader and its variant source in the attack."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from eiger.attacks.corpus_precomputed import CorpusPrecomputedAttack, _cached_engineering_index
from eiger.core.exceptions import IngestionError
from eiger.core.models import AttackConfig, Document
from eiger.datasets import get_dataset
from eiger.datasets.corpus_claim_engineering import (
    CorpusClaimEngineeringDataset,
    CorpusClaimEngineeringGenQDataset,
    load_engineering_rows,
)
from eiger.ingestion.corpus_builder import CorpusBuilder

FIELDS = ["claim_id", "clean_claim", "poisoned_claim", "ground_truth_label", "language",
          "sensitivity_class", "risk_level", "manipulation_category", "source_platform",
          "requires_human_review"]


def _row(cid: str, clean: str, poisoned: str, **over: str) -> dict[str, str]:
    base = {"claim_id": cid, "clean_claim": clean, "poisoned_claim": poisoned,
            "ground_truth_label": "verified_true", "language": "en", "sensitivity_class": "S2",
            "risk_level": "5", "manipulation_category": "M04", "source_platform": "Snopes",
            "requires_human_review": "False"}
    base.update(over)
    return base


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    rows = [
        _row("E1", "Heat caused the cattle deaths in 2022.", "The cattle deaths caused heat in 2022."),
        _row("E2", "False one.", "Falser one.", ground_truth_label="verified_false"),
        _row("E3", "Hola.", "Adios.", language="es"),
        _row("E4", "Sensitive.", "Very sensitive.", sensitivity_class="S3"),
        _row("E5", "Same text.", "Same text."),
        _row("", "", ""),
    ]
    path = tmp_path / "rag_claims_engineering.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _cached_engineering_index.cache_clear()
    return path


def test_filters_and_forces_unreviewed(csv_path: Path) -> None:
    ds = CorpusClaimEngineeringDataset(path=csv_path)
    assert ds.content_hash == "0" * 16
    claims = ds.load()
    assert [c.claim_id for c in claims] == ["E1"]
    meta = claims[0].metadata
    # The file says False, but nothing in this corpus was human-reviewed.
    assert meta["requires_human_review"] is True and meta["variant_source"] == "engineering"
    assert claims[0].risk_level == 5 and len(ds.content_hash) == 16
    assert ds.load(max_claims=0) == []


def test_genq_hash_and_numeric_leak(csv_path: Path, tmp_path: Path) -> None:
    sha = hashlib.sha256(b"Heat caused the cattle deaths in 2022.").hexdigest()
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"item_id": "E1", "context_query": "What caused the cattle deaths?",
                              "original_sha256": sha}]))
    claims = CorpusClaimEngineeringGenQDataset(path=csv_path, query_file=q).load()
    assert [(c.claim_id, c.metadata["query_source"]) for c in claims] == [("E1", "generated")]
    q.write_text(json.dumps([{"item_id": "E1", "context_query": "x", "original_sha256": "0" * 64}]))
    assert CorpusClaimEngineeringGenQDataset(path=csv_path, query_file=q).load() == []
    q.write_text(json.dumps([]))
    assert CorpusClaimEngineeringGenQDataset(path=csv_path, query_file=q).load() == []


def test_genq_excludes_numeric_leak(tmp_path: Path) -> None:
    path = tmp_path / "e.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(_row("N1", "It happened in 2020.", "It happened in 2021.", manipulation_category="M02"))
    sha = hashlib.sha256(b"It happened in 2020.").hexdigest()
    q = tmp_path / "q.json"
    q.write_text(json.dumps([{"item_id": "N1", "context_query": "Did it happen in 2020?", "original_sha256": sha}]))
    assert CorpusClaimEngineeringGenQDataset(path=path, query_file=q).load() == []


def test_errors_and_registry(csv_path: Path, tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="not found"):
        load_engineering_rows(tmp_path / "missing.csv")
    dup = tmp_path / "dup.csv"
    with dup.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([_row("D", "a", "b"), _row("D", "a", "b")])
    with pytest.raises(IngestionError, match="Duplicate"):
        load_engineering_rows(dup)
    CorpusClaimEngineeringDataset().download(str(csv_path.parent))
    with pytest.raises(IngestionError, match="rag_claims_engineering"):
        CorpusClaimEngineeringDataset().download(str(tmp_path / "nowhere"))
    assert get_dataset("corpus_claim_engineering_genq").use_generated_queries is True


def test_attack_engineering_source(csv_path: Path) -> None:
    doc = Document(doc_id="gt_E1", claim_id="E1", text="Heat caused the cattle deaths in 2022.",
                   doc_type="ground_truth")
    attack = CorpusPrecomputedAttack()
    poisoned = attack.apply(doc, seed=1, variant_source="engineering", engineering_file=str(csv_path))
    assert poisoned.text == "The cattle deaths caused heat in 2022."
    assert poisoned.attack_params["variant_source"] == "engineering"
    with pytest.raises(IngestionError, match="Unknown variant_source"):
        attack.apply(doc, seed=1, variant_source="other")


def test_real_engineering_corpus_builds() -> None:
    _cached_engineering_index.cache_clear()
    claims = CorpusClaimEngineeringDataset().load(max_claims=40)
    builder = CorpusBuilder([(CorpusPrecomputedAttack(), AttackConfig(
        name="corpus_precomputed", poison_rate=1.0, params={"variant_source": "engineering"}))])
    result = builder.build(claims)
    assert len(result.poisoned_docs) == 40 and result.skipped_noop == {}


def test_genq_all_and_extra_variants(csv_path: Path, tmp_path: Path) -> None:
    from eiger.datasets.corpus_claim_engineering import (
        CorpusClaimEngineeringExtraGenQDataset,
        CorpusClaimEngineeringGenQAllDataset,
    )

    # E1 is a normal row; N1 repeats a changed year in its query.
    path = tmp_path / "e.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(_row("E1", "Heat caused the cattle deaths in 2022.", "The cattle deaths caused heat in 2022."))
        writer.writerow(_row("N1", "It happened in 2020.", "It happened in 2021.", manipulation_category="M02"))
        writer.writerow(_row("X1", "Storms hit the coast in March.", "Storms hit the coast in April.", manipulation_category="M02"))
    sha = lambda t: hashlib.sha256(t.encode()).hexdigest()  # noqa: E731
    q = tmp_path / "q.json"
    q.write_text(json.dumps([
        {"item_id": "E1", "context_query": "What caused the cattle deaths?", "original_sha256": sha("Heat caused the cattle deaths in 2022.")},
        {"item_id": "N1", "context_query": "Did it happen in 2020?", "original_sha256": sha("It happened in 2020.")},
        {"item_id": "X1", "context_query": "What happened on the coast?", "original_sha256": sha("Storms hit the coast in March.")},
    ]))
    keep = {c.claim_id: c for c in CorpusClaimEngineeringGenQAllDataset(path=path, query_file=q).load()}
    assert set(keep) == {"E1", "N1", "X1"}
    assert keep["N1"].metadata["query_numeric_leak"] is True and keep["E1"].metadata["query_numeric_leak"] is False
    assert {c.claim_id for c in CorpusClaimEngineeringGenQDataset(path=path, query_file=q).load()} == {"E1", "X1"}

    # Extra = claims not in the official corpus: put E1 in a fake official corpus.
    official = tmp_path / "official" / "Original_Claims"
    official.mkdir(parents=True)
    (official / "Original_Corpus_01.csv").write_text("item_id\nE1\n", encoding="utf-8")
    extra = CorpusClaimEngineeringExtraGenQDataset(
        path=path, query_file=q, official_corpus_dir=tmp_path / "official")
    # E1 is in the (fake) official corpus, N1 repeats a changed year: only X1 is left.
    assert [c.claim_id for c in extra.load()] == ["X1"]
    assert extra.name == "corpus_claim_engineering_extra_genq"
