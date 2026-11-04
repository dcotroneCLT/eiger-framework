"""
Generate neutral retrieval queries for the official corpus (corpus_claim_official/).

Why
---
The default query for this corpus is the template "Is it true that <claim>?",
which contains the ground-truth text and gives the true document an
artificial rank-1 advantage (docs/EXPERIMENTS.md §4.5). This script asks the
local LLM for one natural question per claim, with EXACTLY the prompt used for
Snopes (scripts/enrich_snopes_claims.py), so the two experiments use the same
query method.

Leak check
----------
A question must not favour either document. For each claim the script
compares the original and the manipulated text and collects the tokens that
differ (e.g. "50" vs "55", "2022" vs "2021"). If the question contains a token
present only in the original, it leaks the truth; if it contains a token
present only in the manipulated text, it leaks the poison. Both are recorded,
and the ``*_genq`` datasets skip flagged rows.

Run once, on one machine, then commit the output so every machine uses the
same queries (generation depends on the model and Ollama version, which are
recorded in every entry):

    python scripts/generate_corpus_queries.py --limit 20 -o /tmp/pilot_queries.json
    python scripts/generate_corpus_queries.py
    python scripts/generate_corpus_queries.py --sample 60 --sample-out review_queries_sample.xlsx

Output (default): corpus_claim_official/derived/context_queries_v1.json
The file is resumable: existing item_ids are skipped and progress is saved
every --checkpoint-every claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eiger.datasets.corpus_claim_official import (  # noqa: E402
    DEFAULT_CORPUS_DIR,
    DEFAULT_QUERY_FILE,
    CorpusClaimOfficialUnreviewedDataset,
    load_modified_index,
)

# Identical to scripts/enrich_snopes_claims.py::_QUESTION_PROMPT_TEMPLATE.
# Keep them identical: the point is that both corpora use the same method.
QUESTION_PROMPT_TEMPLATE = (
    "You will be given a single verified factual statement. Write ONE natural "
    "English question that this statement directly and fully answers. "
    "Reply with ONLY the question, no preamble, no quotation marks, no numbering.\n\n"
    "Statement: {claim}\n\n"
    "Question:"
)
PROMPT_SHA256 = hashlib.sha256(QUESTION_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?", re.IGNORECASE)
_STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or but not no is are was were be been "
    "it its this that these those as than then there their they he she his her we you i "
    "has have had do does did will would can could may might should about into over".split()
)


def tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)} - _STOPWORDS


def leak_tokens(question: str, original: str, modified: str) -> tuple[list[str], list[str]]:
    """Return (truth-only tokens, poison-only tokens) that appear in the question."""
    q, o, m = tokens(question), tokens(original), tokens(modified)
    return sorted(q & (o - m)), sorted(q & (m - o))


def clean_question(raw: str) -> str:
    return raw.strip().strip('"').strip("'").strip()


def build_entry(item: dict[str, str], question: str, model: str, ollama_version: str) -> dict[str, Any]:
    truth, poison = leak_tokens(question, item["original"], item["modified"])
    return {
        "item_id": item["item_id"],
        "context_query": question,
        "leaks_truth_tokens": truth,
        "leaks_poison_tokens": poison,
        "model": model,
        "ollama_version": ollama_version,
        "prompt_sha256": PROMPT_SHA256,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def candidate_items(corpus_dir: Path) -> list[dict[str, str]]:
    """Every row the official datasets can use, in the datasets' own order."""
    claims = CorpusClaimOfficialUnreviewedDataset(corpus_dir=corpus_dir).load()
    modified = load_modified_index(corpus_dir)
    return [
        {
            "item_id": c.claim_id,
            "original": c.original_fact,
            "modified": modified[c.claim_id]["claim_text"],
            "category": modified[c.claim_id].get("manipulation_category", ""),
            "changed_element": modified[c.claim_id].get("changed_element", ""),
            "platform": str(c.metadata.get("source_platform", "")),
            "review": str(c.metadata.get("human_review_status", "")),
        }
        for c in claims
    ]


def _save(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted(entries, key=lambda e: e["item_id"])
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def generate(
    items: list[dict[str, str]],
    output: Path,
    llm: Any,
    model: str,
    ollama_version: str,
    limit: int | None,
    checkpoint_every: int,
) -> list[dict[str, Any]]:
    existing: list[dict[str, Any]] = (
        json.loads(output.read_text(encoding="utf-8")) if output.is_file() else []
    )
    done = {e["item_id"] for e in existing}
    todo = [it for it in items if it["item_id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    for i, item in enumerate(todo, start=1):
        prompt = QUESTION_PROMPT_TEMPLATE.format(claim=item["original"])
        question = clean_question(llm.generate(prompt))
        existing.append(build_entry(item, question, model, ollama_version))
        if i % checkpoint_every == 0:
            _save(output, existing)
            print(f"  {i}/{len(todo)} generated")
    _save(output, existing)
    return existing


def stratified_sample(
    items: list[dict[str, str]], entries: list[dict[str, Any]], n: int, seed: int
) -> list[dict[str, Any]]:
    """Sample n rows spread across manipulation category, then platform."""
    by_id = {e["item_id"]: e for e in entries}
    pool = [it | {"entry": by_id[it["item_id"]]} for it in items if it["item_id"] in by_id]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        groups[(row["category"], row["platform"])].append(row)
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)
    keys = sorted(groups)
    picked: list[dict[str, Any]] = []
    while len(picked) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop())
    return picked


def write_review_sheet(rows: list[dict[str, Any]], path: Path) -> None:
    import openpyxl
    from openpyxl.styles import Alignment, Font

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "revisione_query"
    header = [
        "item_id", "categoria", "piattaforma", "revisione_claim", "claim_originale",
        "claim_manipolato", "elemento_modificato", "domanda_generata",
        "token_vero_nella_domanda", "token_manipolato_nella_domanda",
        "domanda_ok (si/no)", "problema (neutra? risponde al claim? lingua?)", "note",
    ]
    ws.append(header)
    for r in rows:
        e = r["entry"]
        ws.append([
            r["item_id"], r["category"], r["platform"], r["review"], r["original"],
            r["modified"], r["changed_element"], e["context_query"],
            ", ".join(e["leaks_truth_tokens"]), ", ".join(e["leaks_poison_tokens"]), "", "", "",
        ])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    widths = [12, 9, 11, 22, 60, 60, 18, 50, 16, 16, 14, 30, 30]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "B2"

    info = wb.create_sheet("istruzioni")
    for line in [
        "Revisione delle domande generate per il corpus ufficiale.",
        "Per ogni riga: la domanda è neutra (non contiene né il valore vero né quello manipolato)?",
        "La domanda chiede proprio ciò che il claim afferma? È in inglese corretto?",
        "Compilare 'domanda_ok' con si/no e, se no, descrivere il problema.",
        f"Prompt (SHA-256): {PROMPT_SHA256}",
    ]:
        info.append([line])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_QUERY_FILE)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="generate at most N new queries")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--sample", type=int, default=0, help="also export N rows for human review")
    parser.add_argument("--sample-out", type=Path, default=Path("review_queries_sample.xlsx"))
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--no-generate", action="store_true", help="only export the sample")
    args = parser.parse_args(argv)

    items = candidate_items(args.corpus_dir)
    if args.no_generate:
        entries = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        from eiger.config.settings import get_settings
        from eiger.llm.ollama import OllamaLLM
        import httpx

        settings = get_settings()
        base = f"http://{settings.ollama_host}:{settings.ollama_port}"
        version = httpx.get(f"{base}/api/version", timeout=10).json().get("version", "unknown")
        llm = OllamaLLM(model_name=args.model, host=settings.ollama_host,
                        port=settings.ollama_port, temperature=0.0, max_tokens=64)
        print(f"{len(items)} candidate claims; model {args.model}, Ollama {version}")
        entries = generate(items, args.output, llm, args.model, version,
                           args.limit, args.checkpoint_every)

    n_leak = sum(1 for e in entries if e["leaks_truth_tokens"] or e["leaks_poison_tokens"])
    print(f"{len(entries)} queries in {args.output}; {n_leak} flagged as leaking a changed token")
    print(f"sha256: {hashlib.sha256(args.output.read_bytes()).hexdigest()}")

    if args.sample:
        rows = stratified_sample(items, entries, args.sample, args.sample_seed)
        write_review_sheet(rows, args.sample_out)
        print(f"Review sample of {len(rows)} rows written to {args.sample_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
