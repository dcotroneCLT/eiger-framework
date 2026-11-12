"""
Sensitivity of the poison-rate sweep to queries that repeat a changed number or year.

A query that repeats the TRUE year or number is closer to the original document
than to the manipulated one, so it favours the original (docs/EXPERIMENTS.md).
The default ``*_genq`` datasets drop such claims. This script uses a sweep run
with a ``*_genq_all`` dataset (which keeps them and flags them in
``claim.metadata['query_numeric_leak']``) and compares the two groups.

Reported per group (leak / no leak), pooled over the poisoned rates (> 0%):

- own variant retrieved: share of records whose own poisoned variant is in the top-k.
  The poisoning of a claim does not depend on its query, so this should be similar
  in the two groups.
- variant outranks truth: among records where the own variant IS retrieved, the
  share where it is ranked above the true document. This is the measure of query
  neutrality: a leaking query should lower it.
- PRD@1: share of records with a poisoned document at rank 1.

Usage:
    python scripts/analyze_query_leak_sensitivity.py \\
        --config-dir experiments/corpus_sweep_official_genq_all_preliminary \\
        --results-dir results/corpus_sweep_official_genq_all_preliminary
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUN_RE = re.compile(r"^(?P<rate>\d+)pct_seed(?P<seed>\d+)$")


def classify_record(record: dict) -> dict[str, bool]:
    """Own-variant retrieval, whether it outranks the truth, and poison at rank 1."""
    claim_id = record["claim_id"]
    hits = (record.get("retrieval", {}) or {}).get("hits", []) or []
    own_poison = next(
        (h for h in hits if _doc(h).get("doc_type") == "poisoned" and _doc(h).get("claim_id") == claim_id), None
    )
    own_truth = next(
        (h for h in hits if _doc(h).get("doc_type") == "ground_truth" and _doc(h).get("claim_id") == claim_id), None
    )
    rank1 = next((h for h in hits if h.get("rank") == 1), None)
    return {
        "own_retrieved": own_poison is not None,
        # A variant retrieved while the truth is not in the top-k also outranks it.
        "outranks_truth": own_poison is not None
        and (own_truth is None or own_poison["rank"] < own_truth["rank"]),
        "poison_rank1": rank1 is not None and _doc(rank1).get("doc_type") == "poisoned",
    }


def _doc(hit: dict) -> dict:
    return hit.get("document", {}) or {}


def stratify(results_dir: Path, flags: dict[str, bool | None]) -> dict[str, dict[str, float]]:
    """Pool every run with rate > 0 and aggregate the classification by leak flag."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for run_dir in sorted(results_dir.iterdir()):
        match = RUN_RE.match(run_dir.name)
        results = run_dir / "results.json"
        if not match or int(match["rate"]) == 0 or not results.exists():
            continue
        for record in json.loads(results.read_text(encoding="utf-8"))["records"]:
            flag = flags.get(record["claim_id"])
            if flag is None:
                continue
            group = "leak" if flag else "no_leak"
            info = classify_record(record)
            counts[group]["records"] += 1
            counts[group]["own_retrieved"] += info["own_retrieved"]
            counts[group]["outranks_truth"] += info["outranks_truth"]
            counts[group]["poison_rank1"] += info["poison_rank1"]
    out: dict[str, dict[str, float]] = {}
    for group, c in counts.items():
        out[group] = {
            "records": c["records"],
            "own_retrieved": c["own_retrieved"] / c["records"],
            "outranks_truth_given_retrieved": (
                c["outranks_truth"] / c["own_retrieved"] if c["own_retrieved"] else float("nan")
            ),
            "n_retrieved": c["own_retrieved"],
            "poison_rank1": c["poison_rank1"] / c["records"],
        }
    return out


def leak_flags(dataset_name: str, max_claims: int | None) -> dict[str, bool | None]:
    from eiger.datasets import get_dataset

    claims = get_dataset(dataset_name).load(max_claims=max_claims)
    return {c.claim_id: c.metadata.get("query_numeric_leak") for c in claims}


def main(argv: list[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    config_file = next(iter(sorted(args.config_dir.glob("*.yaml"))), None)
    if config_file is None:
        print(f"No configs in {args.config_dir}", file=sys.stderr)
        return 1
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))["dataset"]
    flags = leak_flags(cfg["name"], cfg.get("max_claims"))
    if not any(v is not None for v in flags.values()):
        print(f"Dataset '{cfg['name']}' has no query_numeric_leak flags: use a *_genq_all dataset.", file=sys.stderr)
        return 1

    table = stratify(args.results_dir, flags)
    n_leak = sum(1 for v in flags.values() if v)
    print(f"# Numeric-leak sensitivity: {cfg['name']}\n")
    print(f"Claims in the sample: {len(flags)}; with a query repeating a changed number or year: {n_leak}.\n")
    print("| Group | Records | Own variant retrieved | Variant outranks truth (given retrieved) | PRD@1 |")
    print("|---|---|---|---|---|")
    for group in ("no_leak", "leak"):
        if group in table:
            t = table[group]
            print(f"| {group} | {t['records']} | {t['own_retrieved']:.3f} | "
                  f"{t['outranks_truth_given_retrieved']:.3f} (n={t['n_retrieved']}) | {t['poison_rank1']:.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
