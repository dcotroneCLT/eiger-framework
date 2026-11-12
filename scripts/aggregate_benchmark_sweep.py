"""
Aggregate the paper-facing poison-rate sweep into a summary table.

Reads results/poison_rate_sweep_benchmark/<rate>pct_seed<seed>/results.json
and reports, for each poison rate:

- PRR and PRD (the retrieval-level results): mean and standard deviation
  across seeds, plus a 95% percentile bootstrap confidence interval computed
  on the per-record values pooled across seeds.
- Source Integrity, ERS and FFR as DIAGNOSTICS only: FFR uses the embedding
  proxy scorer and is structurally zero; ERS averages annotations fixed by
  attack design; SI is not calibrated against human judgments. They are
  printed for completeness and must not be reported as results.

Usage:
    python scripts/aggregate_benchmark_sweep.py
    python scripts/aggregate_benchmark_sweep.py --n-boot 10000 --boot-seed 0

Outputs:
    results/poison_rate_sweep_benchmark/summary.csv
    results/poison_rate_sweep_benchmark/summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "poison_rate_sweep_benchmark"
RUN_RE = re.compile(r"^(?P<rate>\d+)pct_seed(?P<seed>\d+)$")
RESULT_METRICS = ["prr", "prd"]
# Decomposition of PRR (computed per record from the hits, not stored metrics):
#   prr_own   = the queried claim's OWN poisoned variant is in the top-k (targeted)
#   prr_other = only poisoned documents of OTHER claims are in the top-k (collateral)
# prr = prr_own + prr_other for every record.
DECOMPOSITION = ["prr_own", "prr_other"]
DIAGNOSTIC_METRICS = ["source_integrity", "ers", "ffr"]


def _is_poisoned(hit: dict) -> bool:
    doc = hit.get("document", {}) or {}
    return doc.get("doc_type") == "poisoned" or bool(doc.get("is_poisoned"))


def per_record_values(record: dict) -> dict[str, float]:
    """PRR/PRD per record: use the stored metric if present, else recompute from hits."""
    metrics = record.get("metrics", {}) or {}
    hits = (record.get("retrieval", {}) or {}).get("hits", []) or []
    out = {}
    out["prr"] = float(metrics["prr"]) if "prr" in metrics else float(any(_is_poisoned(h) for h in hits))
    if "prd" in metrics:
        out["prd"] = float(metrics["prd"])
    else:
        rank1 = next((h for h in hits if h.get("rank") == 1), None)
        out["prd"] = float(rank1 is not None and _is_poisoned(rank1))
    claim_id = record.get("claim_id")
    poisoned = [h for h in hits if _is_poisoned(h)]
    own = any((h.get("document", {}) or {}).get("claim_id") == claim_id for h in poisoned)
    out["prr_own"] = float(own)
    out["prr_other"] = float(bool(poisoned) and not own)
    return out


def bootstrap_ci(values: list[float], n_boot: int, rng: random.Random) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    lo = means[int(0.025 * n_boot)]
    hi = means[min(int(0.975 * n_boot), n_boot - 1)]
    return lo, hi


def claim_seed_matrices(by_seed: dict[int, dict]):
    """
    Per metric, a (seeds x claims) matrix of per-record values, using the claims
    present in every seed. Returns None if records carry no claim_id.
    """
    import numpy as np

    per_seed: list[dict[str, dict[str, float]]] = []
    for seed in sorted(by_seed):
        values: dict[str, dict[str, float]] = {}
        for rec in by_seed[seed].get("records", []):
            if rec.get("claim_id") is None:
                return None
            values[rec["claim_id"]] = per_record_values(rec)
        per_seed.append(values)
    common = set.intersection(*(set(v) for v in per_seed)) if per_seed else set()
    if not common:
        return None
    ids = sorted(common)
    return {k: np.array([[v[c][k] for c in ids] for v in per_seed]) for k in RESULT_METRICS}


def two_way_bootstrap(matrix, n_boot: int, rng) -> tuple[float, float]:
    """
    95% percentile interval for the mean of a (seeds x claims) matrix, resampling
    SEEDS and CLAIMS independently (pigeonhole bootstrap for crossed factors).

    Why: per-record values are not independent. The same claims appear in every
    seed (claim-level dependence), and within one seed the records share one
    corpus, so a claim's exposure depends on which neighbours were poisoned
    (run-level dependence). Resampling both factors keeps both sources of
    variability; resampling records treats neither.
    """
    import numpy as np

    n_seeds, n_claims = matrix.shape
    means = np.empty(n_boot)
    for b in range(n_boot):
        seeds = rng.integers(0, n_seeds, n_seeds)
        claims = rng.integers(0, n_claims, n_claims)
        means[b] = matrix[np.ix_(seeds, claims)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--boot-seed", type=int, default=0)
    parser.add_argument("--results-dir", default=str(RESULTS.relative_to(REPO)),
                        help="sweep results folder, relative to the repo root")
    parser.add_argument("--title", default="Poison-rate sweep, benchmark attacks only (M01, M02, M04, M06)")
    args = parser.parse_args()
    rng = random.Random(args.boot_seed)
    import numpy as np

    np_rng = np.random.default_rng(args.boot_seed)
    results_dir = REPO / args.results_dir

    runs: dict[int, dict[int, dict]] = defaultdict(dict)
    for run_dir in sorted(results_dir.iterdir() if results_dir.exists() else []):
        m = RUN_RE.match(run_dir.name)
        f = run_dir / "results.json"
        if m and f.exists():
            runs[int(m["rate"])][int(m["seed"])] = json.loads(f.read_text(encoding="utf-8"))
    if not runs:
        print(f"No results found under {results_dir}", file=sys.stderr)
        return 1

    # Consistency checks: same commit and same seeds for every rate.
    commits = {r.get("git_commit") for by_seed in runs.values() for r in by_seed.values()}
    seed_sets = {tuple(sorted(by_seed)) for by_seed in runs.values()}
    if len(commits) > 1:
        print(f"WARNING: results come from different git commits: {sorted(commits)}")
    if len(seed_sets) > 1:
        print(f"WARNING: rates do not all have the same seeds: {seed_sets}")

    rows = []
    for rate in sorted(runs):
        by_seed = runs[rate]
        row = {"rate_pct": rate, "n_seeds": len(by_seed), "seeds": " ".join(map(str, sorted(by_seed)))}
        pooled: dict[str, list[float]] = {k: [] for k in RESULT_METRICS + DECOMPOSITION}
        n_records = 0
        for res in by_seed.values():
            for rec in res.get("records", []):
                vals = per_record_values(rec)
                for k in RESULT_METRICS + DECOMPOSITION:
                    pooled[k].append(vals[k])
                n_records += 1
        row["n_records"] = n_records

        for k in RESULT_METRICS + DIAGNOSTIC_METRICS:
            per_seed = [res["aggregate_metrics"][k] for res in by_seed.values() if k in res.get("aggregate_metrics", {})]
            row[f"{k}_mean"] = statistics.fmean(per_seed) if per_seed else float("nan")
            row[f"{k}_sd"] = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
        for k in DECOMPOSITION:
            row[f"{k}_mean"] = statistics.fmean(pooled[k]) if pooled[k] else float("nan")
        matrices = claim_seed_matrices(by_seed)
        if matrices is None:
            print(f"WARNING: rate {rate}%: records without claim_id; claim x seed interval not computed.")
        for k in RESULT_METRICS:
            lo, hi = bootstrap_ci(pooled[k], args.n_boot, rng)
            row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = lo, hi
            lo2, hi2 = (
                two_way_bootstrap(matrices[k], args.n_boot, np_rng)
                if matrices is not None else (float("nan"), float("nan"))
            )
            row[f"{k}_ci2_lo"], row[f"{k}_ci2_hi"] = lo2, hi2
            # The CI uses per-record values; they must reproduce the stored
            # aggregate, otherwise the record-level reading of results.json is wrong.
            pooled_mean = statistics.fmean(pooled[k]) if pooled[k] else float("nan")
            if abs(pooled_mean - row[f"{k}_mean"]) > 1e-6:
                print(f"WARNING: rate {rate}%: per-record {k} mean {pooled_mean:.4f} "
                      f"!= aggregate mean {row[f'{k}_mean']:.4f}; do not use the CI.")
        rows.append(row)

    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# {args.title}",
        "",
        f"Seeds per rate: {rows[0]['seeds']}. Git commit(s): {', '.join(sorted(map(str, commits)))}. "
        f"Bootstrap: {args.n_boot} resamples, seed {args.boot_seed}.",
        "",
        "## Results (retrieval level)",
        "",
        "IC (records) treats every record as independent. IC (claims x seeds) resamples claims and seeds "
        "independently, so it accounts for the same claims appearing in every seed and for the shared corpus "
        "within a run; it is the one to report.",
        "",
        "| Rate | PRR mean ± sd | PRR IC (records) | PRR IC (claims x seeds) | PRD@1 mean ± sd | PRD@1 IC (records) | PRD@1 IC (claims x seeds) | Records |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['rate_pct']}% | {r['prr_mean']:.3f} ± {r['prr_sd']:.3f} | [{r['prr_ci_lo']:.3f}, {r['prr_ci_hi']:.3f}] "
            f"| [{r['prr_ci2_lo']:.3f}, {r['prr_ci2_hi']:.3f}] "
            f"| {r['prd_mean']:.3f} ± {r['prd_sd']:.3f} | [{r['prd_ci_lo']:.3f}, {r['prd_ci_hi']:.3f}] "
            f"| [{r['prd_ci2_lo']:.3f}, {r['prd_ci2_hi']:.3f}] | {r['n_records']} |"
        )
    lines += [
        "",
        "## PRR decomposition",
        "",
        "Own = the queried claim's own poisoned variant is in the top-k (targeted exposure). "
        "Other only = only poisoned documents of other claims are in the top-k (collateral exposure).",
        "",
        "| Rate | PRR | own | other only |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['rate_pct']}% | {r['prr_mean']:.3f} | {r['prr_own_mean']:.3f} | {r['prr_other_mean']:.3f} |")
    lines += [
        "",
        "## Diagnostics (NOT results)",
        "",
        "FFR uses the embedding-similarity proxy and is structurally zero; ERS averages annotations fixed "
        "by attack design; Source Integrity is not calibrated against human judgments.",
        "",
        "| Rate | Source Integrity | ERS | FFR (embedding proxy) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['rate_pct']}% | {r['source_integrity_mean']:.3f} ± {r['source_integrity_sd']:.3f} "
            f"| {r['ers_mean']:.3f} ± {r['ers_sd']:.3f} | {r['ffr_mean']:.3f} |"
        )
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWritten: {(results_dir / 'summary.csv').relative_to(REPO)}, {(results_dir / 'summary.md').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
