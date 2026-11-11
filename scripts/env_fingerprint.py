"""
Environment and data fingerprint for EIBench reproducibility.

Usage:
    python scripts/env_fingerprint.py --out fingerprint_<host>.json
    python scripts/env_fingerprint.py --compare ref.json new.json

Run it on the reference machine (where the paper's numbers were produced) and on
the new machine, then compare. Any line printed by --compare is a difference that
can change results and must be understood before running experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Packages whose version can change numerical results.
KEY_PACKAGES = [
    "eiger", "numpy", "torch", "transformers", "sentence-transformers",
    "qdrant-client", "rank-bm25", "httpx", "pydantic", "structlog",
    "PyYAML", "ragas",
]

# Data files that define the experiment input. Their hashes must match.
DATA_GLOBS = [
    "corpus_claim_official/Original_Claims/*.csv",
    "corpus_claim_official/Modified_Claims/*.csv",
    "corpus_claim/*.csv",
    "experiments/**/*.yaml",
    "eibench_raw_claims.json",
]

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1:8b"


def run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or out.stderr.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def torch_devices() -> dict:
    try:
        import torch  # type: ignore

        return {
            "cuda": torch.cuda.is_available(),
            "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "num_threads": torch.get_num_threads(),
        }
    except Exception as exc:  # torch missing or broken
        return {"error": repr(exc)}


def hf_embedder_revision() -> str | None:
    base = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snaps = list(base.glob(f"models--{EMBEDDER.replace('/', '--')}/snapshots/*"))
    return snaps[0].name if snaps else None


def ollama_info() -> dict:
    version = run(["ollama", "--version"])
    model_digest = None
    listing = run(["ollama", "list"]) or ""
    for line in listing.splitlines():
        if line.startswith(LLM_MODEL):
            parts = line.split()
            model_digest = parts[1] if len(parts) > 1 else None
    return {"version": version, "llm_model": LLM_MODEL, "llm_model_id": model_digest}


def qdrant_info() -> dict:
    out = run(["curl", "-s", "http://localhost:6333/healthz"])
    return {"healthz": out}


def git_info() -> dict:
    commit = run(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    dirty = run(["git", "-C", str(REPO), "status", "--porcelain"])
    return {"commit": commit, "dirty_files": len(dirty.splitlines()) if dirty else 0}


def data_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for pattern in DATA_GLOBS:
        for path in sorted(REPO.glob(pattern)):
            if path.is_file():
                hashes[str(path.relative_to(REPO))] = sha256(path)
    return hashes


def collect() -> dict:
    packages = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name.lower()] = dist.version
    return {
        "git": git_info(),
        "python": {"version": platform.python_version(), "impl": platform.python_implementation()},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),  # arm64 vs x86_64 matters
            "platform": platform.platform(),
        },
        "key_packages": {p: pkg_version(p) for p in KEY_PACKAGES},
        "all_packages": dict(sorted(packages.items())),
        "torch_devices": torch_devices(),
        "embedder": {"model": EMBEDDER, "hf_snapshot": hf_embedder_revision()},
        "ollama": ollama_info(),
        "qdrant": qdrant_info(),
        "data_sha256": data_hashes(),
    }


def compare(a: dict, b: dict, prefix: str = "") -> list[str]:
    diffs: list[str] = []
    keys = set(a) | set(b)
    for k in sorted(keys):
        path = f"{prefix}{k}"
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            diffs += compare(va, vb, prefix=path + ".")
        elif va != vb:
            diffs.append(f"DIFF {path}: reference={va!r}  new={vb!r}")
    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="write fingerprint JSON here")
    parser.add_argument("--compare", nargs=2, metavar=("REF", "NEW"), help="compare two fingerprints")
    args = parser.parse_args()

    if args.compare:
        ref = json.loads(Path(args.compare[0]).read_text())
        new = json.loads(Path(args.compare[1]).read_text())
        diffs = compare(ref, new)
        if not diffs:
            print("OK: environment and data fingerprints are identical.")
            return 0
        print(f"{len(diffs)} difference(s). Resolve before running experiments:")
        print("\n".join(diffs))
        return 1

    fp = collect()
    text = json.dumps(fp, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text)
        print(f"Fingerprint written to {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
