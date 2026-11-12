"""Tests for scripts/analyze_query_leak_sensitivity.py and the claims x seeds bootstrap."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


analysis = _load("analyze_query_leak_sensitivity")
aggregate = _load("aggregate_benchmark_sweep")


def _rec(claim_id: str, hits: list[tuple[int, str, str]]) -> dict:
    return {"claim_id": claim_id, "retrieval": {"hits": [
        {"rank": r, "document": {"doc_type": t, "claim_id": c}} for r, t, c in hits]}}


def test_classify_record() -> None:
    own_above = _rec("A", [(1, "poisoned", "A"), (2, "ground_truth", "A")])
    own_below = _rec("A", [(1, "ground_truth", "A"), (2, "poisoned", "A")])
    other_only = _rec("A", [(1, "ground_truth", "A"), (2, "poisoned", "B")])
    truth_missing = _rec("A", [(1, "ground_truth", "B"), (2, "poisoned", "A")])
    assert analysis.classify_record(own_above) == {"own_retrieved": True, "outranks_truth": True, "poison_rank1": True}
    assert analysis.classify_record(own_below) == {"own_retrieved": True, "outranks_truth": False, "poison_rank1": False}
    assert analysis.classify_record(other_only)["own_retrieved"] is False
    assert analysis.classify_record(truth_missing)["outranks_truth"] is True
    assert analysis.classify_record({"claim_id": "A"})["own_retrieved"] is False


def test_stratify_pools_poisoned_runs_and_splits_by_flag(tmp_path: Path) -> None:
    def write(name: str, records: list[dict]) -> None:
        d = tmp_path / name
        d.mkdir()
        (d / "results.json").write_text(json.dumps({"records": records}))

    write("0pct_seed42", [_rec("A", [(1, "poisoned", "A")])])          # ignored: rate 0
    write("10pct_seed42", [
        _rec("A", [(1, "ground_truth", "A"), (2, "poisoned", "A")]),   # leak, retrieved, below truth
        _rec("B", [(1, "poisoned", "B"), (2, "ground_truth", "B")]),   # no leak, retrieved, above truth
        _rec("C", [(1, "ground_truth", "C")]),                         # no leak, not retrieved
        _rec("Z", [(1, "ground_truth", "Z")]),                         # no flag: skipped
    ])
    (tmp_path / "notes.txt").write_text("not a run")
    (tmp_path / "10pct_seed43").mkdir()                                # run without results.json
    table = analysis.stratify(tmp_path, {"A": True, "B": False, "C": False, "Z": None})
    assert table["leak"]["outranks_truth_given_retrieved"] == 0.0 and table["leak"]["records"] == 1
    assert table["no_leak"]["records"] == 2
    assert table["no_leak"]["own_retrieved"] == 0.5
    assert table["no_leak"]["outranks_truth_given_retrieved"] == 1.0
    assert table["no_leak"]["poison_rank1"] == 0.5


def test_stratify_handles_group_without_retrieved_variants(tmp_path: Path) -> None:
    d = tmp_path / "5pct_seed42"
    d.mkdir()
    (d / "results.json").write_text(json.dumps({"records": [_rec("A", [(1, "ground_truth", "A")])]}))
    table = analysis.stratify(tmp_path, {"A": False})
    assert np.isnan(table["no_leak"]["outranks_truth_given_retrieved"])


def test_main_reports_and_validates_flags(tmp_path: Path, monkeypatch, capsys) -> None:
    cfg_dir, res_dir = tmp_path / "cfg", tmp_path / "res"
    cfg_dir.mkdir()
    res_dir.mkdir()
    assert analysis.main(["--config-dir", str(cfg_dir), "--results-dir", str(res_dir)]) == 1  # no configs
    (cfg_dir / "a.yaml").write_text("dataset: {name: fake, max_claims: 5}\n")
    monkeypatch.setattr(analysis, "leak_flags", lambda name, n: {"A": None})
    assert analysis.main(["--config-dir", str(cfg_dir), "--results-dir", str(res_dir)]) == 1  # no flags
    d = res_dir / "10pct_seed42"
    d.mkdir()
    (d / "results.json").write_text(json.dumps({"records": [_rec("A", [(1, "poisoned", "A")]),
                                                              _rec("B", [(1, "ground_truth", "B")])]}))
    monkeypatch.setattr(analysis, "leak_flags", lambda name, n: {"A": True, "B": False})
    assert analysis.main(["--config-dir", str(cfg_dir), "--results-dir", str(res_dir)]) == 0
    out = capsys.readouterr().out
    assert "with a query repeating a changed number or year: 1" in out and "| leak |" in out


def test_two_way_bootstrap_is_wider_than_record_bootstrap_with_claim_effect() -> None:
    rng = np.random.default_rng(0)
    # 5 seeds x 100 claims; 20 claims are almost always exposed (strong claim effect).
    claim_effect = np.array([0.9 if i < 20 else 0.05 for i in range(100)])
    matrix = (rng.random((5, 100)) < claim_effect).astype(float)
    lo2, hi2 = aggregate.two_way_bootstrap(matrix, 2000, np.random.default_rng(1))
    lo1, hi1 = aggregate.bootstrap_ci(list(matrix.ravel()), 2000, random.Random(1))
    assert (hi2 - lo2) > (hi1 - lo1)
    assert lo2 <= matrix.mean() <= hi2


def test_claim_seed_matrices() -> None:
    def run(claims: list[str]) -> dict:
        return {"records": [_rec(c, [(1, "ground_truth", c), (2, "poisoned", c)]) for c in claims]}

    m = aggregate.claim_seed_matrices({42: run(["A", "B", "C"]), 43: run(["B", "C", "D"])})
    assert m["prr"].shape == (2, 2)            # only claims present in every seed: B, C
    assert aggregate.claim_seed_matrices({42: {"records": [{"metrics": {}, "retrieval": {"hits": []}}]}}) is None
    assert aggregate.claim_seed_matrices({42: {"records": []}}) is None
