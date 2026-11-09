"""Tests for CorpusClaimOfficialDataset(s) and CorpusPrecomputedAttack."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from eiger.attacks import get_attack
from eiger.attacks.corpus_precomputed import CorpusPrecomputedAttack, _cached_index
from eiger.core.exceptions import IngestionError
from eiger.core.models import AttackConfig, Document
from eiger.datasets import get_dataset
from eiger.datasets.corpus_claim_official import (
    DEFAULT_CORPUS_DIR,
    CorpusClaimOfficialDataset,
    CorpusClaimOfficialUnreviewedDataset,
    _parse_risk_level,
    load_modified_index,
)
from eiger.ingestion.corpus_builder import CorpusBuilder

FIELDS = [
    "item_id", "claim_text", "claim_original_reference", "original_normalized_label",
    "language_iso", "final_inclusion", "sensitivity_class", "human_review_status",
    "risk_level", "source_platform", "topic", "manipulation_category", "changed_element",
]


def _row(item_id: str, text: str, **over: str) -> dict[str, str]:
    base = {
        "item_id": item_id, "claim_text": text, "claim_original_reference": text,
        "original_normalized_label": "verified_true", "language_iso": "en",
        "final_inclusion": "TRUE", "sensitivity_class": "S1",
        "human_review_status": "kept_after_manual_audit", "risk_level": "4",
        "source_platform": "Snopes", "topic": "t", "manipulation_category": "",
        "changed_element": "",
    }
    base.update(over)
    return base


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _modified(original: dict[str, str], text: str, category: str = "M01") -> dict[str, str]:
    return _row(original["item_id"], text, claim_original_reference=original["claim_text"],
                manipulation_category=category, changed_element="x",
                human_review_status=original["human_review_status"])


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    ok = _row("A1", "Rate rose to 3% in 2020.")
    unreviewed = _row("A2", "Unreviewed claim.", human_review_status="not_manually_reviewed")
    originals = [
        ok,
        unreviewed,
        _row("B1", "False claim.", original_normalized_label="verified_false"),
        _row("B2", "Frase.", language_iso="it"),
        _row("B3", "Excluded.", final_inclusion="FALSE"),
        _row("B4", "S3 claim.", sensitivity_class="S3"),
        _row("B5", "No pair."),
        _row("B6", "Mismatch.", risk_level="9"),
        _row("", ""),
    ]
    modified = [
        _modified(ok, "Rate rose to 30% in 2020."),
        _modified(unreviewed, "Unreviewed claim, changed.", "M06"),
        {**_modified(originals[7], "Mismatch changed."), "claim_original_reference": "other"},
    ]
    _write(tmp_path / "Original_Claims" / "Original_Corpus_01.csv", originals)
    _write(tmp_path / "Modified_Claims" / "Modified_Corpus_01.csv", modified)
    _cached_index.cache_clear()
    return tmp_path


def test_reviewed_dataset_applies_every_filter(corpus: Path) -> None:
    ds = CorpusClaimOfficialDataset(corpus_dir=corpus)
    assert ds.content_hash == "0" * 16
    claims = ds.load()
    assert [c.claim_id for c in claims] == ["A1"]
    claim = claims[0]
    assert claim.ground_truth_label == "verified_true"
    assert claim.risk_level == 4 and claim.sensitivity_class == "S1"
    assert claim.metadata["manipulation_category"] == "M01"
    assert claim.metadata["requires_human_review"] is False
    assert claim.context_query == "Is it true that Rate rose to 3% in 2020.?"
    assert len(ds.content_hash) == 16


def test_unreviewed_dataset_and_max_claims(corpus: Path) -> None:
    ds = CorpusClaimOfficialUnreviewedDataset(corpus_dir=corpus)
    ids = {c.claim_id for c in ds.load()}
    assert ids == {"A1", "A2"}
    assert len(ds.load(max_claims=1)) == 1
    flagged = [c for c in ds.load() if c.claim_id == "A2"][0]
    assert flagged.metadata["requires_human_review"] is True


def test_registry_names() -> None:
    assert get_dataset("corpus_claim_official").name == "corpus_claim_official"
    assert get_dataset("corpus_claim_official_unreviewed").include_unreviewed is True


def test_missing_files_and_duplicates(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="No files"):
        CorpusClaimOfficialDataset(corpus_dir=tmp_path).load()
    row = _row("D1", "x")
    _write(tmp_path / "Modified_Claims" / "Modified_Corpus_01.csv", [row, row])
    with pytest.raises(IngestionError, match="Duplicate"):
        load_modified_index(tmp_path)


def test_download_guard(corpus: Path, tmp_path: Path) -> None:
    CorpusClaimOfficialDataset().download(str(corpus))
    with pytest.raises(IngestionError, match="Original_Claims"):
        CorpusClaimOfficialDataset().download(str(tmp_path / "nowhere"))


@pytest.mark.parametrize("raw,expected", [("", None), (None, None), ("abc", None), ("0", None), ("6", None), ("3", 3)])
def test_parse_risk_level(raw: str | None, expected: int | None) -> None:
    assert _parse_risk_level(raw) == expected


def _gt(claim_id: str, text: str) -> Document:
    return Document(doc_id=f"gt_{claim_id}", claim_id=claim_id, text=text, doc_type="ground_truth",
                    sensitivity_class="S1", risk_level=4, ground_truth_label="verified_true")


def test_attack_returns_corpus_variant(corpus: Path) -> None:
    attack = get_attack("corpus_precomputed")
    poisoned = attack.apply(_gt("A1", "Rate rose to 3% in 2020."), seed=1, corpus_dir=str(corpus))
    assert poisoned.text == "Rate rose to 30% in 2020."
    assert poisoned.original_text == "Rate rose to 3% in 2020."
    assert poisoned.attack_params["manipulation_category"] == "M01"
    assert poisoned.attack_params["no_op"] is False
    assert poisoned.annotation is None
    assert poisoned.sensitivity_class == "S1"


def test_attack_integrity_errors(corpus: Path) -> None:
    attack = CorpusPrecomputedAttack()
    with pytest.raises(IngestionError, match="No manipulated variant"):
        attack.apply(_gt("ZZ", "x"), seed=1, corpus_dir=str(corpus))
    with pytest.raises(IngestionError, match="does not reference"):
        attack.apply(_gt("A1", "different text"), seed=1, corpus_dir=str(corpus))


def test_attack_refuses_m03_m05(tmp_path: Path) -> None:
    orig = _row("F1", "Source said X.")
    _write(tmp_path / "Modified_Claims" / "Modified_Corpus_01.csv", [_modified(orig, "Blog said X.", "M03")])
    _cached_index.cache_clear()
    with pytest.raises(IngestionError, match="M03"):
        CorpusPrecomputedAttack().apply(_gt("F1", "Source said X."), seed=1, corpus_dir=str(tmp_path))


def test_real_tracked_corpus_builds_with_default_dir() -> None:
    """The corpus tracked in git loads and every selected claim has a valid poison."""
    _cached_index.cache_clear()
    claims = CorpusClaimOfficialUnreviewedDataset().load(max_claims=50)
    assert len(claims) == 50
    builder = CorpusBuilder([(CorpusPrecomputedAttack(), AttackConfig(name="corpus_precomputed", poison_rate=1.0))])
    result = builder.build(claims)
    assert len(result.poisoned_docs) == 50
    assert all(p.text != p.original_text for p in result.poisoned_docs)
    assert DEFAULT_CORPUS_DIR.name == "corpus_claim_official"


# ─── Generated queries (*_genq datasets) ──────────────────────────────────────

import json  # noqa: E402

from eiger.datasets.corpus_claim_official import (  # noqa: E402
    CorpusClaimOfficialGenQDataset,
    CorpusClaimOfficialUnreviewedGenQDataset,
    load_query_index,
)


def _queries(path: Path, entries: list[dict]) -> Path:
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def _h(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_genq_uses_generated_query_and_applies_checks(corpus: Path, tmp_path: Path) -> None:
    qfile = _queries(tmp_path / "q.json", [
        # Neutral: shares "2020" with both texts, repeats neither "3" nor "30".
        {"item_id": "A1", "context_query": "How much did the rate rise in 2020?",
         "original_sha256": _h("Rate rose to 3% in 2020.")},
        # Repeats "changed", present only in the manipulated text -> leak.
        {"item_id": "A2", "context_query": "Was the claim changed?",
         "original_sha256": _h("Unreviewed claim.")},
    ])
    reviewed = CorpusClaimOfficialGenQDataset(corpus_dir=corpus, query_file=qfile).load()
    assert [(c.claim_id, c.context_query, c.metadata["query_source"]) for c in reviewed] == [
        ("A1", "How much did the rate rise in 2020?", "generated")
    ]
    unreviewed = CorpusClaimOfficialUnreviewedGenQDataset(corpus_dir=corpus, query_file=qfile).load()
    assert [c.claim_id for c in unreviewed] == ["A1"]


def test_genq_skips_missing_query_and_stale_hash(corpus: Path, tmp_path: Path) -> None:
    only_a2 = _queries(tmp_path / "q2.json", [{"item_id": "A2", "context_query": "q"}])
    assert CorpusClaimOfficialGenQDataset(corpus_dir=corpus, query_file=only_a2).load() == []
    stale = _queries(tmp_path / "q3.json", [
        {"item_id": "A1", "context_query": "What happened to the rate?", "original_sha256": "0" * 64}
    ])
    assert CorpusClaimOfficialGenQDataset(corpus_dir=corpus, query_file=stale).load() == []


def test_template_variant_marks_query_source(corpus: Path) -> None:
    claim = CorpusClaimOfficialDataset(corpus_dir=corpus).load()[0]
    assert claim.metadata["query_source"] == "template"


def test_missing_query_file_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="generate_corpus_queries"):
        load_query_index(tmp_path / "absent.json")
