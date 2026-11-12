"""
CorpusClaimEngineeringDataset — loader for the larger claim corpus in
``corpus_claim/rag_claims_engineering.csv``.

What this corpus is
--------------------
4,620 clean/poisoned claim pairs produced by the collaborator's Mistral
pipeline, WITHOUT the manual corrections of the official corpus
(``corpus_claim_official/``). It contains every official claim plus 982
more; for the 230 official rows that were corrected by hand
(``corrected_after_manual_audit``) the manipulated text here is the original
Mistral output. The corpus author confirmed (2026-11-06) that it may be used
when a larger corpus is needed, noting that some manipulations could have
been done better. Its main value is coverage: 53 verified-true English M04
claims, against 2 in the official corpus.

Review status
--------------
The file's ``requires_human_review`` column is ``False`` for every row, but
these rows were not reviewed by a human. That column is therefore IGNORED:
every claim is marked ``requires_human_review = True`` and every result must
be labelled preliminary. There is no reviewed-only variant.

Filters, ordering and queries are the same as in
``eiger.datasets.corpus_claim_official`` (verified-true, English, not S3,
non-empty manipulation that differs from the original; ordering by SHA-256 of
the claim id; optional generated queries checked by text hash and numeric
leak). Poisoned documents come from the ``corpus_precomputed`` attack with
``params: {variant_source: engineering}``.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from eiger.core.exceptions import IngestionError
from eiger.core.interfaces import BaseDataset
from eiger.core.models import Claim
from eiger.datasets.corpus_claim_official import (
    _ORIGINAL_GLOB,
    DEFAULT_CORPUS_DIR,
    DEFAULT_QUERY_FILE,
    _parse_risk_level,
    _read_csv_dir,
    _sha,
    load_query_index,
)
from eiger.utils.logging import get_logger
from eiger.utils.query_leak import numeric_leak_tokens

log = get_logger(__name__)

DEFAULT_ENGINEERING_FILE = (
    Path(__file__).resolve().parents[2] / "corpus_claim" / "rag_claims_engineering.csv"
)

_METADATA_COLUMNS = ("source_platform", "factcheck_url", "topic", "subtopic", "source_version")


def load_engineering_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise IngestionError(f"Engineering corpus '{path}' not found (tracked in git under corpus_claim/).")
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    ids = [r.get("claim_id", "") for r in rows]
    if len(ids) != len(set(ids)):
        raise IngestionError(f"Duplicate claim_id in '{path}'.")
    return rows


def load_engineering_variant_index(path: Path) -> dict[str, dict[str, str]]:
    """claim_id -> row in the shape used by the corpus_precomputed attack."""
    return {
        r["claim_id"]: {
            "claim_text": r.get("poisoned_claim", ""),
            "claim_original_reference": r.get("clean_claim", ""),
            "manipulation_category": r.get("manipulation_category", ""),
            "changed_element": "",
            "human_review_status": "mistral_only_not_reviewed",
        }
        for r in load_engineering_rows(path)
    }


class CorpusClaimEngineeringDataset(BaseDataset):
    """Verified-true English rows of the larger Mistral-only corpus (preliminary)."""

    name: str = "corpus_claim_engineering"
    description: str = (
        "ENGINEERING ONLY: larger Mistral-only corpus (corpus_claim/), not human-"
        "reviewed. Results are preliminary."
    )
    use_generated_queries: bool = False
    keep_numeric_leaks: bool = False
    # Independent replication: only claims that are NOT in the official corpus.
    only_not_in_official: bool = False

    def __init__(
        self,
        path: str | Path | None = None,
        query_file: str | Path | None = None,
        official_corpus_dir: str | Path | None = None,
    ) -> None:
        self.official_corpus_dir = (
            Path(official_corpus_dir) if official_corpus_dir is not None else DEFAULT_CORPUS_DIR
        )
        self.path = Path(path) if path is not None else DEFAULT_ENGINEERING_FILE
        self.query_file = Path(query_file) if query_file is not None else DEFAULT_QUERY_FILE
        self._loaded_claims: list[Claim] = []

    def download(self, target_dir: str) -> None:
        """No-op guard: the corpus is versioned in git, not downloaded."""
        if not (Path(target_dir) / "rag_claims_engineering.csv").is_file():
            raise IngestionError(f"'{target_dir}' has no rag_claims_engineering.csv.")

    def load(self, split: str = "test", max_claims: int | None = None) -> list[Claim]:
        """``split`` is accepted for interface compatibility and ignored."""
        rows = load_engineering_rows(self.path)
        queries = load_query_index(self.query_file) if self.use_generated_queries else {}
        official_ids = (
            {r["item_id"] for r in _read_csv_dir(self.official_corpus_dir, _ORIGINAL_GLOB)}
            if self.only_not_in_official
            else set()
        )
        skipped: dict[str, int] = {}

        def skip(reason: str) -> None:
            skipped[reason] = skipped.get(reason, 0) + 1

        claims: list[Claim] = []
        for row in rows:
            cid, text, poison = row.get("claim_id", ""), row.get("clean_claim", ""), row.get("poisoned_claim", "")
            if not cid or not text:
                skip("blank")
            elif cid in official_ids:
                skip("in_official_corpus")
            elif row.get("ground_truth_label") != "verified_true":
                skip("not_verified_true")
            elif row.get("language") != "en":
                skip("not_english")
            elif row.get("sensitivity_class") == "S3":
                skip("sensitivity_S3")
            elif not poison.strip() or poison.strip() == text.strip():
                skip("no_effective_manipulation")
            elif self.use_generated_queries and cid not in queries:
                skip("no_generated_query")
            elif self.use_generated_queries and queries[cid].get("original_sha256") != _sha(text):
                skip("generated_query_for_different_text")
            elif (
                self.use_generated_queries
                and not self.keep_numeric_leaks
                and numeric_leak_tokens(queries[cid]["context_query"], text, poison)
            ):
                skip("generated_query_leaks_changed_number")
            else:
                query = queries[cid]["context_query"] if self.use_generated_queries else None
                leak = bool(numeric_leak_tokens(query, text, poison)) if query is not None else None
                claims.append(self._to_claim(row, query, leak))

        claims.sort(key=lambda c: hashlib.sha256(c.claim_id.encode("utf-8")).hexdigest())
        if max_claims is not None:
            claims = claims[:max_claims]
        self._loaded_claims = claims
        log.info("corpus_claim_engineering.load_complete", dataset=self.name,
                 n_claims=len(claims), skipped=skipped)
        return claims

    @property
    def content_hash(self) -> str:
        if not self._loaded_claims:
            return "0" * 16
        combined = "".join(c.claim_id + c.original_fact for c in self._loaded_claims)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    def _to_claim(self, row: dict[str, str], query: str | None, leak: bool | None = None) -> Claim:
        text = row["clean_claim"]
        sensitivity = row.get("sensitivity_class")
        metadata: dict[str, Any] = {k: row[k] for k in _METADATA_COLUMNS if row.get(k)}
        metadata.update(
            manipulation_category=row.get("manipulation_category", ""),
            variant_source="engineering",
            # The file says False for every row, but nothing here was reviewed.
            requires_human_review=True,
            query_source="generated" if query is not None else "template",
            query_numeric_leak=leak,
        )
        return Claim(
            claim_id=row["claim_id"],
            original_fact=text,
            context_query=query if query is not None else f"Is it true that {text}?",
            source_dataset=self.name,
            sensitivity_class=sensitivity if sensitivity in ("S0", "S1", "S2") else None,
            risk_level=_parse_risk_level(row.get("risk_level")),
            ground_truth_label="verified_true",
            metadata=metadata,
        )


class CorpusClaimEngineeringGenQDataset(CorpusClaimEngineeringDataset):
    """Larger corpus with neutral generated queries. Engineering use only."""

    name: str = "corpus_claim_engineering_genq"
    description: str = (
        "ENGINEERING ONLY: larger Mistral-only corpus with neutral generated "
        "queries. Results are preliminary."
    )
    use_generated_queries: bool = True


class CorpusClaimEngineeringGenQAllDataset(CorpusClaimEngineeringDataset):
    """Sensitivity analysis: larger corpus, generated queries, numeric-leak rows kept."""

    name: str = "corpus_claim_engineering_genq_all"
    description: str = (
        "ENGINEERING ONLY: larger Mistral-only corpus, generated queries, "
        "numeric-leak rows kept (flagged in metadata). Results are preliminary."
    )
    use_generated_queries: bool = True
    keep_numeric_leaks: bool = True


class CorpusClaimEngineeringExtraGenQDataset(CorpusClaimEngineeringDataset):
    """
    Independent replication set: the claims of the larger corpus that are NOT
    in the official corpus, with generated queries. Disjoint from every claim
    used in the official-corpus sweeps, so agreement with them is not an
    artefact of shared claims.
    """

    name: str = "corpus_claim_engineering_extra_genq"
    description: str = (
        "ENGINEERING ONLY: claims present only in the larger Mistral-only corpus "
        "(disjoint from the official corpus), generated queries. Results are preliminary."
    )
    use_generated_queries: bool = True
    only_not_in_official: bool = True
