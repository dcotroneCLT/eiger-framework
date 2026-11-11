"""Tests for scripts/generate_corpus_queries.py (no Ollama needed)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import openpyxl
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "generate_corpus_queries", Path(__file__).resolve().parents[2] / "scripts" / "generate_corpus_queries.py"
)
gcq = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gcq)  # type: ignore[union-attr]


def test_prompt_is_identical_to_snopes() -> None:
    spec = importlib.util.spec_from_file_location(
        "enrich", Path(__file__).resolve().parents[2] / "scripts" / "enrich_snopes_claims.py"
    )
    enrich = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enrich)  # type: ignore[union-attr]
    assert gcq.QUESTION_PROMPT_TEMPLATE == enrich._QUESTION_PROMPT_TEMPLATE


def test_leak_tokens_detects_truth_and_poison() -> None:
    orig, mod = "Donated $50 million in 2022.", "Donated $55 million in 2021."
    assert gcq.leak_tokens("How much was donated?", orig, mod) == ([], [])
    assert gcq.leak_tokens("Was $50 million donated in 2021?", orig, mod) == (["50"], ["2021"])


class _FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return '  "How much was donated?"  '


def _items() -> list[dict[str, str]]:
    return [
        {"item_id": f"I{i}", "original": f"Donated {i}0 dollars.", "modified": f"Donated {i}5 dollars.",
         "category": "M01" if i % 2 else "M06", "changed_element": "x",
         "platform": "Snopes" if i < 3 else "PolitiFact", "review": "not_manually_reviewed"}
        for i in range(6)
    ]


def test_generate_is_resumable_and_records_provenance(tmp_path: Path) -> None:
    out, llm = tmp_path / "q.json", _FakeLLM()
    gcq.generate(_items(), out, llm, "m", "0.1", limit=2, checkpoint_every=1, version="v2")
    gcq.generate(_items(), out, llm, "m", "0.1", limit=None, checkpoint_every=100, version="v2")
    entries = json.loads(out.read_text())
    assert llm.calls == 6 and len(entries) == 6
    first = entries[0]
    assert first["context_query"] == "How much was donated?"
    assert first["prompt_version"] == "v2" and first["prompt_sha256"] == gcq.prompt_sha256("v2")
    assert first["manipulation_category"] in {"M01", "M06"}
    assert len(first["original_sha256"]) == 64 and len(first["modified_sha256"]) == 64


def test_generate_refuses_mixing_prompt_versions(tmp_path: Path) -> None:
    out = tmp_path / "q.json"
    gcq.generate(_items(), out, _FakeLLM(), "m", "0.1", limit=1, checkpoint_every=1, version="v1")
    with pytest.raises(SystemExit, match="different file"):
        gcq.generate(_items(), out, _FakeLLM(), "m", "0.1", limit=1, checkpoint_every=1, version="v2")


def test_v2_prompt_is_default_and_asks_for_neutral_question() -> None:
    assert gcq.DEFAULT_PROMPT_VERSION == "v2"
    v2 = gcq.PROMPTS["v2"]
    assert "do not assume that the statement is true" in v2
    assert "Do not include any number" in v2


def test_sample_is_stratified_and_sheet_is_written(tmp_path: Path) -> None:
    entries = [gcq.build_entry(it, "How much?", "m", "v") for it in _items()]
    rows = gcq.stratified_sample(_items(), entries, n=4, seed=0)
    assert len(rows) == 4
    assert len({(r["category"], r["platform"]) for r in rows}) == 4
    path = tmp_path / "s.xlsx"
    gcq.write_review_sheet(rows, path)
    ws = openpyxl.load_workbook(path)["revisione_query"]
    assert ws.max_row == 5 and ws["H2"].value == "How much?" and ws["I2"].value == "v1"
    assert ws["M1"].value.startswith("presuppone")
