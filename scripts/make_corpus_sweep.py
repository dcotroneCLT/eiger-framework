"""
Generate the poison-rate sweep configs for the official corpus in git.

Dataset: ``corpus_claim_official_unreviewed`` (verified-true, English rows of
corpus_claim_official/, human review NOT required). Results are therefore
PRELIMINARY: the corpus is not yet human-validated (alignment protocol,
point 6). When enough rows are reviewed, rerun with ``--reviewed-only``,
which switches to the gated ``corpus_claim_official`` dataset.

Poison: the ``corpus_precomputed`` attack, i.e. the collaborator's own
manipulated claim (M01/M02/M04/M06). Each claim has exactly one variant, so
the nominal poison rate equals the expected share of poisoned claims.

The claim subsample is fixed (ordered by SHA-256 of item_id); the seed only
changes which claims receive their poisoned variant.

Usage:
    python scripts/make_corpus_sweep.py --seeds 42 43 44 45 46 --max-claims 300
    python scripts/make_corpus_sweep.py --dry-run

Output: experiments/corpus_sweep_preliminary/<rate>pct_seed<seed>.yaml
        (or experiments/corpus_sweep_reviewed/ with --reviewed-only)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
RATES = [0, 1, 3, 5, 10]


DATASETS = {
    # (corpus, queries, reviewed_only) -> dataset name
    ("official", "template", False): "corpus_claim_official_unreviewed",
    ("official", "template", True): "corpus_claim_official",
    ("official", "genq", False): "corpus_claim_official_unreviewed_genq",
    ("official", "genq", True): "corpus_claim_official_genq",
    ("engineering", "template", False): "corpus_claim_engineering",
    ("engineering", "genq", False): "corpus_claim_engineering_genq",
    # Sensitivity analysis: keep queries that repeat a changed number or year.
    ("official", "genq_all", False): "corpus_claim_official_unreviewed_genq_all",
    ("engineering", "genq_all", False): "corpus_claim_engineering_genq_all",
    # Independent replication: claims that are only in the larger corpus.
    ("engineering_extra", "genq", False): "corpus_claim_engineering_extra_genq",
}

# Which corpus file the corpus_precomputed attack reads its manipulations from.
VARIANT_SOURCE = {"official": "official", "engineering": "engineering", "engineering_extra": "engineering"}


RETRIEVERS = ("dense", "sparse", "hybrid")


def sweep_name(corpus: str, queries: str, reviewed_only: bool, retriever: str = "dense") -> str:
    kind = "reviewed" if reviewed_only else "preliminary"
    # Dense sweeps keep their original folder names, so existing results stay valid.
    if retriever == "dense":
        if corpus == "official" and queries == "template":
            return f"corpus_sweep_{kind}"
        return f"corpus_sweep_{corpus}_{queries}_{kind}"
    return f"corpus_sweep_{corpus}_{queries}_{retriever}_{kind}"


def build_config(
    rate_pct: int, seed: int, max_claims: int, reviewed_only: bool,
    corpus: str = "official", queries: str = "template", retriever: str = "dense",
) -> dict:
    kind = "reviewed" if reviewed_only else "preliminary"
    name = sweep_name(corpus, queries, reviewed_only, retriever)
    tag = f"{rate_pct}pct_seed{seed}"
    attacks = (
        [{"name": "corpus_precomputed", "poison_rate": rate_pct / 100,
          "params": {"variant_source": VARIANT_SOURCE[corpus]}}]
        if rate_pct > 0 else []
    )
    return {
        "experiment_id": f"{name}_{tag}",
        "description": (
            f"{corpus} corpus sweep ({kind}, {queries} queries, {retriever} retrieval), rate={rate_pct}%, seed={seed}, "
            "collaborator's precomputed manipulations (M01/M02/M04/M06)."
            + ("" if reviewed_only else " PRELIMINARY: corpus not yet human-validated.")
        ),
        "seed": seed,
        "dataset": {
            "name": DATASETS[(corpus, queries, reviewed_only)],
            "path": None,
            "split": "test",
            "max_claims": max_claims,
        },
        "allow_non_benchmark_attacks": False,
        "attacks": attacks,
        "retriever": {
            "type": retriever,
            "embedder": "sentence-transformers/all-MiniLM-L6-v2",
            "vector_store": "qdrant",
            "top_k": 5,
            "collection_name": f"eibench_{name}_{tag}",
        },
        "llm": {"backend": "ollama", "model": "llama3.1:8b", "temperature": 0.0, "max_tokens": 512},
        "faithfulness_scorer": "embedding",
        "metrics": ["ffr", "source_integrity", "ers", "prr", "prd"],
        "output_dir": f"results/{name}/{tag}/",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--max-claims", type=int, default=300)
    parser.add_argument("--reviewed-only", action="store_true")
    parser.add_argument("--retriever", choices=RETRIEVERS, default="dense",
                        help="retrieval method; sparse = BM25, hybrid = dense + BM25 fused with RRF")
    parser.add_argument("--corpus", choices=["official", "engineering", "engineering_extra"],
                        default="official",
                        help="engineering_extra = claims only in the larger corpus (disjoint from official)")
    parser.add_argument("--queries", choices=["template", "genq", "genq_all"], default="template",
                        help="template 'Is it true that ...?'; genq = neutral generated queries (v2, "
                             "numeric-leak rows excluded); genq_all = same, leak rows kept (sensitivity)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.corpus != "official" and args.reviewed_only:
        parser.error("only the official corpus has human-reviewed rows")
    if (args.corpus, args.queries, args.reviewed_only) not in DATASETS:
        parser.error(f"combination not available: corpus={args.corpus}, queries={args.queries}")
    out_dir = REPO / "experiments" / sweep_name(args.corpus, args.queries, args.reviewed_only, args.retriever)
    planned = [
        (out_dir / f"{rate}pct_seed{seed}.yaml",
         build_config(rate, seed, args.max_claims, args.reviewed_only, args.corpus, args.queries, args.retriever))
        for rate in RATES
        for seed in args.seeds
    ]
    for path, cfg in planned:
        print(f"{path.relative_to(REPO)}  dataset={cfg['dataset']['name']}  "
              f"max_claims={args.max_claims}  attacks={[a['name'] for a in cfg['attacks']]}")
    if args.dry_run:
        print(f"\n{len(planned)} configs planned (dry run, nothing written).")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "# GENERATED by scripts/make_corpus_sweep.py — do not edit by hand.\n"
        f"# Corpus: {args.corpus}; queries: {args.queries}.\n"
        + ("" if args.reviewed_only else "# PRELIMINARY: corpus not yet human-validated.\n")
    )
    for path, cfg in planned:
        path.write_text(header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"\nWrote {len(planned)} configs to {out_dir.relative_to(REPO)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
