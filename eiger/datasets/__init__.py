"""
Dataset loaders for EIBench experiments.

This package exposes the dataset registry (register_dataset, get_dataset,
list_datasets) and every concrete BaseDataset implementation, mirroring
the pattern already established by eiger.attacks and eiger.metrics.

Currently registered datasets
------------------------------
  - JSONFixtureDataset ("json_fixture") — bundled development/CI fixture.
  - SnopesDataset ("snopes") — LLM-enriched Snopes fact-checks (True-rated
    subset only). Requires running scripts/enrich_snopes_claims.py first;
    see eiger.datasets.snopes for the full pipeline and scripts/README.md.
  - AVeriTecDataset ("averitec") — AVeriTeC fact-checks (Supported-label
    subset only), using each record's own evidence questions as
    context_query (no LLM enrichment needed). Requires pre-downloaded
    *.jsonl split files under data/averitec/ — see eiger.datasets.averitec
    and docs/DATASETS.md section 3 for manual download steps (automated
    download() is not yet implemented).
  - PolitiFactDataset ("politifact") — LIAR statements ("true"-label
    subset only), with a templated context_query fallback (no LLM
    needed). Requires pre-downloaded *.tsv split files under
    data/politifact/ — see eiger.datasets.politifact and
    docs/DATASETS.md section 4 for manual download steps (automated
    download() is not yet implemented).
  - FactCheckDataset ("factcheck_org") — FactCheck.org fact-checks via
    the CheckThat! mirror ("true"-verdict subset only), with a templated
    context_query fallback (no LLM needed). Requires pre-downloaded
    *.jsonl split files under data/factcheck/ — see eiger.datasets.factcheck
    and docs/DATASETS.md section 5 for manual download steps (automated
    download() is not yet implemented).
  - CorpusClaimDataset ("corpus_claim") — the project's human-curated,
    Mistral-generated claim corpus (5,672 rows, already labeled
    verified_true/verified_false). requires_human_review is True for
    every row today, so load() returns zero claims unless
    include_unreviewed=True is passed explicitly (engineering use only)
    — see eiger.datasets.corpus_claim and
    docs/CLAIM_AND_RESEARCH_QUESTIONS.md §7. Requires the workbook to be
    copied/symlinked to data/corpus_claim/ (no automated download — this
    is not a public corpus).

See docs/DATASETS.md for the full dataset roadmap. All five originally
documented datasets, plus the human-curated corpus_claim loader, now have
implemented loaders.

Responsibilities of this module
--------------------------------
- Re-export the registry API so callers never import
  eiger.datasets.registry directly.
- Re-export every built-in dataset class for direct import convenience.
- Trigger auto-registration of all built-in datasets at import time.

What this module does NOT do
-----------------------------
- It does not call .load() or .download() on anything; construction and
  invocation are the caller's responsibility (typically resolved from a
  DatasetConfig.name via get_dataset()).
- It does not implement AVeriTecDataset's/PolitiFactDataset's/
  FactCheckDataset's automated download(); all three loaders' download()
  is a guard (raises if data is missing) rather than a fetcher — see
  their module docstrings.
"""

# ─── Registry helpers ───────────────────────────────────────────────────────

# ─── Built-in dataset classes ────────────────────────────────────────────────
from eiger.datasets.averitec import AVeriTecDataset
from eiger.datasets.corpus_claim import CorpusClaimDataset
from eiger.datasets.corpus_claim_official import (
    CorpusClaimOfficialDataset,
    CorpusClaimOfficialGenQDataset,
    CorpusClaimOfficialUnreviewedDataset,
    CorpusClaimOfficialUnreviewedGenQDataset,
)
from eiger.datasets.factcheck import FactCheckDataset
from eiger.datasets.json_fixture import JSONFixtureDataset
from eiger.datasets.politifact import PolitiFactDataset
from eiger.datasets.registry import get_dataset, list_datasets, register_dataset
from eiger.datasets.snopes import SnopesDataset

# ─── Auto-registration ───────────────────────────────────────────────────────

# Registered at import time so that `import eiger.datasets` immediately
# enables `get_dataset("json_fixture")` / `get_dataset("snopes")` /
# `get_dataset("averitec")` / `get_dataset("politifact")` /
# `get_dataset("factcheck_org")` with no further setup. Idempotent, so
# repeated imports (e.g. across test modules) are harmless.
register_dataset(JSONFixtureDataset)
register_dataset(SnopesDataset)
register_dataset(AVeriTecDataset)
register_dataset(PolitiFactDataset)
register_dataset(FactCheckDataset)
register_dataset(CorpusClaimDataset)
register_dataset(CorpusClaimOfficialDataset)
register_dataset(CorpusClaimOfficialUnreviewedDataset)
register_dataset(CorpusClaimOfficialGenQDataset)
register_dataset(CorpusClaimOfficialUnreviewedGenQDataset)

# ─── Public API declaration ───────────────────────────────────────────────────

__all__ = [
    # Registry interface
    "register_dataset",
    "get_dataset",
    "list_datasets",
    # Concrete dataset classes
    "JSONFixtureDataset",
    "SnopesDataset",
    "AVeriTecDataset",
    "PolitiFactDataset",
    "FactCheckDataset",
    "CorpusClaimDataset",
    "CorpusClaimOfficialDataset",
    "CorpusClaimOfficialUnreviewedDataset",
    "CorpusClaimOfficialGenQDataset",
    "CorpusClaimOfficialUnreviewedGenQDataset",
]
