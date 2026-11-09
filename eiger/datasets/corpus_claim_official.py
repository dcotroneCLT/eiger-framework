"""
CorpusClaimOfficialDataset — loader for the official claim corpus tracked
in git under ``corpus_claim_official/``.

Where the data comes from
--------------------------
The collaborator's corpus is versioned in this repository as two paired sets
of CSV files with the same 34-column schema and the same ``item_id`` order:

- ``corpus_claim_official/Original_Claims/Original_Corpus_*.csv``: the
  unmodified fact-checked claims (``is_poisoned = FALSE``).
- ``corpus_claim_official/Modified_Claims/Modified_Corpus_*.csv``: the
  collaborator's controlled manipulation of each claim (``is_poisoned =
  TRUE``), with its category (M01/M02/M04/M06) and changed element.

This loader reads the original side. The modified side is consumed by the
``corpus_precomputed`` attack (``eiger.attacks.corpus_precomputed``), so the
poisoned documents in an experiment are exactly the collaborator's reviewed
manipulations, not EIGER's mechanical attacks.

Which rows become Claims
-------------------------
- ``original_normalized_label == "verified_true"`` only. The ground-truth
  document of an experiment is the claim text itself, and every metric that
  compares an answer against ground truth assumes that text is true. A
  ``verified_false`` claim would need a corrective document, which this
  corpus does not contain, so those rows are left out rather than
  misused as ground truth.
- ``language_iso == "en"`` only: the default embedder is English-only.
- ``final_inclusion == TRUE`` and ``sensitivity_class != "S3"``.
- A paired row must exist in ``Modified_Claims`` with the same ``item_id``
  and ``claim_original_reference`` equal to the original text; otherwise
  the poisoned variant could not be traced back to its source.

Human-review gate
------------------
The corpus records per-row review in ``human_review_status``. Only
``kept_after_manual_audit`` and ``approved_final_version`` mean a human
accepted the final row. ``corrected_after_manual_audit`` rows are still
``manual_corrected_pending_output_recheck`` in ``final_qc_status``, so they
are NOT treated as reviewed.

- ``corpus_claim_official`` (this class) returns reviewed rows only. It is
  the only variant that may feed paper-facing results.
- ``corpus_claim_official_unreviewed`` returns every row that passes the
  other filters. It exists for engineering runs, and every result produced
  with it must be labelled "preliminary, corpus not yet validated", as the
  alignment protocol requires.

Queries
--------
By default the query is the template ``Is it true that <claim>?``. That
template contains the ground-truth text, which gives the true document an
artificial rank-1 advantage (see docs/EXPERIMENTS.md §4.5). The ``*_genq``
variants instead read neutral questions generated once by
``scripts/generate_corpus_queries.py`` and versioned in
``corpus_claim_official/derived/``. Each entry records the SHA-256 of the
claim text it was generated from; an entry whose hash no longer matches the
claim is not used. The leak check is recomputed here, against the
manipulated variant actually used: a question that repeats a token differing
between the original and the manipulated claim is skipped, because it would
favour one of the two documents.

Ordering and ``max_claims``
----------------------------
The CSV files are sorted by source (all Snopes rows first). Taking the first
N rows would therefore bias any subsample. Claims are instead ordered by the
SHA-256 of ``item_id``, a fixed, seed-independent permutation, so
``max_claims=N`` always returns the same mixed subsample on every machine.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from eiger.core.exceptions import IngestionError
from eiger.core.interfaces import BaseDataset
from eiger.core.models import Claim
from eiger.utils.logging import get_logger
from eiger.utils.query_leak import leak_tokens

log = get_logger(__name__)

# Repository root is parents[2] of eiger/datasets/corpus_claim_official.py.
DEFAULT_CORPUS_DIR = Path(__file__).resolve().parents[2] / "corpus_claim_official"

_ORIGINAL_GLOB = "Original_Claims/Original_Corpus_*.csv"
_MODIFIED_GLOB = "Modified_Claims/Modified_Corpus_*.csv"

DEFAULT_QUERY_FILE = DEFAULT_CORPUS_DIR / "derived" / "context_queries_v2.json"

REVIEWED_STATUSES = frozenset({"kept_after_manual_audit", "approved_final_version"})

_METADATA_COLUMNS = (
    "source_platform",
    "factcheck_url",
    "original_rating",
    "topic",
    "subtopic",
    "human_review_status",
    "final_qc_status",
    "source_version",
    "schema_version",
)


def _read_csv_dir(corpus_dir: Path, pattern: str) -> list[dict[str, str]]:
    files = sorted(corpus_dir.glob(pattern))
    if not files:
        raise IngestionError(
            f"No files matching '{pattern}' under '{corpus_dir}'. The official "
            "corpus is tracked in git under corpus_claim_official/."
        )
    rows: list[dict[str, str]] = []
    for path in files:
        with path.open(encoding="utf-8", newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def load_modified_index(corpus_dir: Path) -> dict[str, dict[str, str]]:
    """Map item_id -> modified row. Raises on duplicate item_ids."""
    index: dict[str, dict[str, str]] = {}
    for row in _read_csv_dir(corpus_dir, _MODIFIED_GLOB):
        item_id = row.get("item_id", "")
        if item_id in index:
            raise IngestionError(f"Duplicate item_id '{item_id}' in Modified_Claims.")
        index[item_id] = row
    return index


def load_query_index(path: Path) -> dict[str, dict[str, Any]]:
    """Map item_id -> generated-query entry from a generate_corpus_queries.py file."""
    if not path.is_file():
        raise IngestionError(
            f"Generated query file '{path}' not found. Create it once with "
            "scripts/generate_corpus_queries.py and commit it, so every machine "
            "uses the same queries."
        )
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {entry["item_id"]: entry for entry in entries}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_risk_level(raw: str | None) -> int | None:
    try:
        value = int(raw) if raw not in (None, "") else None
    except ValueError:
        return None
    return value if value is not None and 1 <= value <= 5 else None


class CorpusClaimOfficialDataset(BaseDataset):
    """Human-reviewed, verified-true, English rows of the official corpus."""

    name: str = "corpus_claim_official"
    description: str = (
        "Official claim corpus tracked in git (corpus_claim_official/). "
        "Human-reviewed, verified-true, English rows only; poisoned variants "
        "come from the 'corpus_precomputed' attack."
    )
    include_unreviewed: bool = False
    use_generated_queries: bool = False

    def __init__(
        self, corpus_dir: str | Path | None = None, query_file: str | Path | None = None
    ) -> None:
        self.corpus_dir = Path(corpus_dir) if corpus_dir is not None else DEFAULT_CORPUS_DIR
        self.query_file = Path(query_file) if query_file is not None else DEFAULT_QUERY_FILE
        self._loaded_claims: list[Claim] = []

    def download(self, target_dir: str) -> None:
        """No-op guard: the corpus is versioned in git, not downloaded."""
        if not (Path(target_dir) / "Original_Claims").is_dir():
            raise IngestionError(
                f"'{target_dir}' has no Original_Claims/ folder. The official "
                "corpus is tracked in git under corpus_claim_official/."
            )

    def load(self, split: str = "test", max_claims: int | None = None) -> list[Claim]:
        """
        Load claims. ``split`` is accepted for interface compatibility and
        ignored: the corpus has no train/test partition.
        """
        originals = _read_csv_dir(self.corpus_dir, _ORIGINAL_GLOB)
        modified = load_modified_index(self.corpus_dir)
        queries = load_query_index(self.query_file) if self.use_generated_queries else {}

        skipped: dict[str, int] = {}

        def skip(reason: str) -> None:
            skipped[reason] = skipped.get(reason, 0) + 1

        claims: list[Claim] = []
        for row in originals:
            item_id = row.get("item_id", "")
            text = row.get("claim_text", "")
            if not item_id or not text:
                skip("blank")
            elif row.get("original_normalized_label") != "verified_true":
                skip("not_verified_true")
            elif row.get("language_iso") != "en":
                skip("not_english")
            elif row.get("final_inclusion") != "TRUE":
                skip("not_final_inclusion")
            elif row.get("sensitivity_class") == "S3":
                skip("sensitivity_S3")
            elif item_id not in modified:
                skip("no_modified_pair")
            elif modified[item_id].get("claim_original_reference") != text:
                skip("modified_pair_text_mismatch")
            elif not self.include_unreviewed and row.get("human_review_status") not in REVIEWED_STATUSES:
                skip("not_human_reviewed")
            elif self.use_generated_queries and item_id not in queries:
                skip("no_generated_query")
            elif self.use_generated_queries and queries[item_id].get("original_sha256") != _sha(text):
                skip("generated_query_for_different_text")
            elif self.use_generated_queries and any(
                leak_tokens(queries[item_id]["context_query"], text, modified[item_id]["claim_text"])
            ):
                skip("generated_query_leaks_changed_tokens")
            else:
                query = queries[item_id]["context_query"] if self.use_generated_queries else None
                claims.append(self._to_claim(row, modified[item_id], query))

        claims.sort(key=lambda c: hashlib.sha256(c.claim_id.encode("utf-8")).hexdigest())
        if max_claims is not None:
            claims = claims[:max_claims]

        self._loaded_claims = claims
        log.info(
            "corpus_claim_official.load_complete",
            dataset=self.name,
            n_claims=len(claims),
            include_unreviewed=self.include_unreviewed,
            skipped=skipped,
        )
        return claims

    @property
    def content_hash(self) -> str:
        if not self._loaded_claims:
            return "0" * 16
        combined = "".join(c.claim_id + c.original_fact for c in self._loaded_claims)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    def _to_claim(
        self, row: dict[str, str], modified_row: dict[str, str], query: str | None = None
    ) -> Claim:
        text = row["claim_text"]
        sensitivity = row.get("sensitivity_class")
        metadata: dict[str, Any] = {k: row[k] for k in _METADATA_COLUMNS if row.get(k)}
        metadata["manipulation_category"] = modified_row.get("manipulation_category", "")
        metadata["corpus_dir"] = str(self.corpus_dir)
        metadata["requires_human_review"] = row.get("human_review_status") not in REVIEWED_STATUSES
        metadata["query_source"] = "generated" if query is not None else "template"
        return Claim(
            claim_id=row["item_id"],
            original_fact=text,
            context_query=query if query is not None else f"Is it true that {text}?",
            source_dataset=self.name,
            sensitivity_class=sensitivity if sensitivity in ("S0", "S1", "S2") else None,
            risk_level=_parse_risk_level(row.get("risk_level")),
            ground_truth_label="verified_true",
            metadata=metadata,
        )


class CorpusClaimOfficialUnreviewedDataset(CorpusClaimOfficialDataset):
    """Same filters, without the human-review gate. Engineering use only."""

    name: str = "corpus_claim_official_unreviewed"
    description: str = (
        "ENGINEERING ONLY: official corpus including rows not yet human-"
        "reviewed. Results must be labelled 'preliminary, corpus not yet "
        "validated'."
    )
    include_unreviewed: bool = True


class CorpusClaimOfficialGenQDataset(CorpusClaimOfficialDataset):
    """Reviewed rows, with neutral generated queries instead of the template."""

    name: str = "corpus_claim_official_genq"
    description: str = (
        "Official corpus, human-reviewed rows, neutral generated queries "
        "(corpus_claim_official/derived/)."
    )
    use_generated_queries: bool = True


class CorpusClaimOfficialUnreviewedGenQDataset(CorpusClaimOfficialDataset):
    """All rows, neutral generated queries. Engineering use only."""

    name: str = "corpus_claim_official_unreviewed_genq"
    description: str = (
        "ENGINEERING ONLY: official corpus including unreviewed rows, with "
        "neutral generated queries. Results are preliminary."
    )
    include_unreviewed: bool = True
    use_generated_queries: bool = True
