"""
CorpusPrecomputedAttack — uses the collaborator's own manipulated claim as
the poisoned document.

Why this exists
----------------
EIGER's mechanical attacks (numerical_shift, date_manipulation, ...) apply
rule-based edits. The official corpus (``corpus_claim_official/``) already
contains, for every claim, a controlled manipulation produced and quality-
checked in the collaborator's pipeline, with its category (M01/M02/M04/M06)
and the exact element changed. This attack looks that variant up instead of
generating one, so experiments on the official corpus measure the
manipulations the corpus actually defines.

How it fits CorpusBuilder
--------------------------
Nothing changes in CorpusBuilder: it still decides, per claim and with the
experiment seed, whether the claim is poisoned (``poison_rate``), and calls
``apply()`` only for the selected claims. ``apply()`` itself is
deterministic and ignores ``seed``: there is exactly one manipulated variant
per claim.

Integrity checks
-----------------
``apply()`` refuses (``IngestionError``) when no modified row exists for the
claim, or when the modified row's ``claim_original_reference`` does not equal
the ground-truth text. Either case means the poisoned document could not be
traced back to its source, which the ethics protocol forbids.

Variant source
---------------
``params: {variant_source: official}`` (default) reads
``corpus_claim_official/Modified_Claims`` (manually curated). ``engineering``
reads ``corpus_claim/rag_claims_engineering.csv`` (Mistral-only, larger).

ERS
----
The corpus has per-row plausibility/editorial-risk columns, but they are
still empty pending human annotation. ``annotation`` is therefore ``None``
and ERS will be 0.0 for these documents. That is deliberate: inventing
annotation values would make ERS meaningless.

Benchmark status
-----------------
Only M01/M02/M04/M06 exist in the corpus (M03/M05 have zero rows), so this
attack is not ``excluded_from_benchmark``. As a safety net, ``apply()``
refuses any row whose category is M03 or M05.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from eiger.core.exceptions import IngestionError
from eiger.core.interfaces import BaseAttack
from eiger.core.models import Document, PoisonedDocument
from eiger.datasets.corpus_claim_engineering import (
    DEFAULT_ENGINEERING_FILE,
    load_engineering_variant_index,
)
from eiger.datasets.corpus_claim_official import DEFAULT_CORPUS_DIR, load_modified_index

_FORBIDDEN_CATEGORIES = frozenset({"M03", "M05"})


@lru_cache(maxsize=4)
def _cached_index(corpus_dir: str) -> dict[str, dict[str, str]]:
    return load_modified_index(Path(corpus_dir))


@lru_cache(maxsize=4)
def _cached_engineering_index(path: str) -> dict[str, dict[str, str]]:
    return load_engineering_variant_index(Path(path))


class CorpusPrecomputedAttack(BaseAttack):
    """Poison with the corpus's own pre-generated manipulation."""

    name: str = "corpus_precomputed"
    description: str = (
        "Uses the manipulated claim from corpus_claim_official/Modified_Claims "
        "(collaborator's pipeline, M01/M02/M04/M06) as the poisoned document."
    )
    excluded_from_benchmark: bool = False

    def apply(self, document: Document, seed: int, **kwargs: Any) -> PoisonedDocument:
        variant_source = kwargs.get("variant_source", "official")
        if variant_source == "official":
            location = str(Path(kwargs.get("corpus_dir") or DEFAULT_CORPUS_DIR).resolve())
            row = _cached_index(location).get(document.claim_id)
        elif variant_source == "engineering":
            location = str(Path(kwargs.get("engineering_file") or DEFAULT_ENGINEERING_FILE).resolve())
            row = _cached_engineering_index(location).get(document.claim_id)
        else:
            raise IngestionError(
                f"Unknown variant_source '{variant_source}' (expected 'official' or 'engineering')."
            )
        if row is None:
            raise IngestionError(
                f"No manipulated variant for claim '{document.claim_id}' in {location}."
            )
        if row.get("claim_original_reference") != document.text:
            raise IngestionError(
                f"Manipulated variant for claim '{document.claim_id}' does not "
                "reference the ground-truth text; refusing an untraceable poison."
            )
        category = row.get("manipulation_category", "")
        if category in _FORBIDDEN_CATEGORIES:
            raise IngestionError(
                f"Claim '{document.claim_id}' has category {category}, excluded "
                "from the benchmark by the ethics protocol."
            )

        return PoisonedDocument(
            doc_id=document.doc_id,
            claim_id=document.claim_id,
            text=row["claim_text"],
            attack_name=self.name,
            attack_params=self.describe()
            | {
                "manipulation_category": category,
                "changed_element": row.get("changed_element", ""),
                "human_review_status": row.get("human_review_status", ""),
                "no_op": row["claim_text"] == document.text,
                "variant_source": variant_source,
            },
            original_text=document.text,
            annotation=None,
            sensitivity_class=document.sensitivity_class,
            risk_level=document.risk_level,
            ground_truth_label=document.ground_truth_label,
        )

    def describe(self) -> dict[str, Any]:
        return {"attack": self.name, "method": "lookup in corpus_claim_official/Modified_Claims"}
