"""
Generate neutral retrieval queries for the official corpus (corpus_claim_official/).

Why
---
The default query for this corpus is the template "Is it true that <claim>?",
which contains the ground-truth text and gives the true document an
artificial rank-1 advantage (docs/EXPERIMENTS.md §4.5).

Prompt versions
---------------
- v1: identical to the Snopes prompt (scripts/enrich_snopes_claims.py). With
  it, about half of the questions repeated the true value (year, number,
  omitted phrase), mostly as yes/no rephrasings of the claim.
- v2 (default): agreed with the corpus author on 2026-11-04. A natural
  rephrasing that does not contain the specific values that a manipulation
  could change, and that does not assume the statement is true (some claims
  are rated false, and every manipulated claim is inaccurate).

Each entry records the prompt version and its SHA-256, the model, the Ollama
version and the SHA-256 of the original and manipulated claim texts, so every
query stays linked to the exact claim pair it was generated for.

Leak check
----------
Recomputed by the ``*_genq`` datasets at load time (eiger/utils/query_leak.py)
against the manipulated variant actually used; the flags stored here are
informational.

Run once, on one machine, then commit the output so every machine uses the
same queries:

    python scripts/generate_corpus_queries.py --limit 20 -o /tmp/pilot_v2.json
    python scripts/generate_corpus_queries.py
    python scripts/generate_corpus_queries.py --no-generate --sample 60 --sample-out review_queries_v2.xlsx

Output (default): corpus_claim_official/derived/context_queries_v2.json
The file is resumable: existing item_ids are skipped and progress is saved
every --checkpoint-every claims. Mixing prompt versions in one file is refused.
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
from eiger.datasets.corpus_claim_engineering import (  # noqa: E402
    CorpusClaimEngineeringDataset,
    load_engineering_variant_index,
    DEFAULT_ENGINEERING_FILE,
)
from eiger.utils.query_leak import leak_tokens, numeric_leak_tokens, tokens  # noqa: E402,F401

# v1 is identical to scripts/enrich_snopes_claims.py::_QUESTION_PROMPT_TEMPLATE.
PROMPTS = {
    "v1": (
        "You will be given a single verified factual statement. Write ONE natural "
        "English question that this statement directly and fully answers. "
        "Reply with ONLY the question, no preamble, no quotation marks, no numbering.\n\n"
        "Statement: {claim}\n\n"
        "Question:"
    ),
    "v2": (
        "You will be given a statement taken from a fact-checking archive. The statement "
        "may be accurate or inaccurate. Write ONE natural English question that a "
        "journalist would ask to find out what actually happened about the subject of "
        "the statement. Rephrase in your own words instead of copying the statement. "
        "Do not include any number, quantity, percentage, date or year mentioned in the "
        "statement, and do not assume that the statement is true. "
        "Reply with ONLY the question, no preamble, no quotation marks, no numbering.\n\n"
        "Statement: {claim}\n\n"
        "Question:"
    ),
}
DEFAULT_PROMPT_VERSION = "v2"
QUESTION_PROMPT_TEMPLATE = PROMPTS["v1"]


def prompt_sha256(version: str) -> str:
    return hashlib.sha256(PROMPTS[version].encode("utf-8")).hexdigest()


PROMPT_SHA256 = prompt_sha256("v1")


def clean_question(raw: str) -> str:
    return raw.strip().strip('"').strip("'").strip()


def build_entry(
    item: dict[str, str], question: str, model: str, ollama_version: str, version: str = "v1"
) -> dict[str, Any]:
    truth, poison = leak_tokens(question, item["original"], item["modified"])
    return {
        "item_id": item["item_id"],
        "context_query": question,
        "manipulation_category": item.get("category", ""),
        "original_sha256": hashlib.sha256(item["original"].encode("utf-8")).hexdigest(),
        "modified_sha256": hashlib.sha256(item["modified"].encode("utf-8")).hexdigest(),
        "leaks_truth_tokens": truth,
        "leaks_poison_tokens": poison,
        "prompt_version": version,
        "prompt_sha256": prompt_sha256(version),
        "model": model,
        "ollama_version": ollama_version,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def candidate_items(corpus_dir: Path, source: str = "official") -> list[dict[str, str]]:
    """
    Every row the datasets of ``source`` can use, in the datasets' own order.

    The two corpora share claim ids and original texts, so one query file
    serves both: generating with ``source="engineering"`` after ``official``
    only adds the claims the larger corpus has in addition.
    """
    if source == "engineering":
        claims = CorpusClaimEngineeringDataset(path=DEFAULT_ENGINEERING_FILE).load()
        modified = load_engineering_variant_index(DEFAULT_ENGINEERING_FILE)
    else:
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
    version: str = "v1",
) -> list[dict[str, Any]]:
    existing: list[dict[str, Any]] = (
        json.loads(output.read_text(encoding="utf-8")) if output.is_file() else []
    )
    other = {e.get("prompt_version", "v1") for e in existing} - {version}
    if other:
        raise SystemExit(
            f"{output} already contains queries from prompt {sorted(other)}; "
            f"write prompt {version} to a different file."
        )
    done = {e["item_id"] for e in existing}
    todo = [it for it in items if it["item_id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    for i, item in enumerate(todo, start=1):
        prompt = PROMPTS[version].format(claim=item["original"])
        question = clean_question(llm.generate(prompt))
        existing.append(build_entry(item, question, model, ollama_version, version))
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
        "claim_manipolato", "elemento_modificato", "domanda_generata", "versione_prompt",
        "token_vero_nella_domanda", "token_manipolato_nella_domanda",
        "domanda_ok (si/no)", "presuppone_che_sia_vero (si/no)",
        "problema (neutra? riformulata? riguarda il claim? lingua?)", "note",
    ]
    ws.append(header)
    for r in rows:
        e = r["entry"]
        ws.append([
            r["item_id"], r["category"], r["platform"], r["review"], r["original"],
            r["modified"], r["changed_element"], e["context_query"],
            e.get("prompt_version", "v1"),
            ", ".join(e["leaks_truth_tokens"]), ", ".join(e["leaks_poison_tokens"]), "", "", "", "",
        ])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    widths = [12, 9, 11, 22, 60, 60, 18, 50, 9, 16, 16, 14, 16, 30, 30]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "B2"

    info = wb.create_sheet("istruzioni")
    for line in [
        "Revisione delle domande generate per il corpus ufficiale.",
        "Ogni riga collega domanda, claim originale, claim manipolato e categoria (item_id).",
        "1. La domanda è neutra? Non deve contenere né il valore vero né quello manipolato.",
        "2. È una riformulazione naturale, non una copia del claim?",
        "3. Riguarda proprio ciò che il claim afferma?",
        "4. Presuppone che il claim sia vero? (non dovrebbe: alcuni claim sono falsi)",
        "Compilare 'domanda_ok' e 'presuppone_che_sia_vero' con si/no; descrivere eventuali problemi.",
        *[f"Prompt {v} (SHA-256): {prompt_sha256(v)}"
          for v in sorted({r["entry"].get("prompt_version", "v1") for r in rows})],
    ]:
        info.append([line])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_QUERY_FILE)
    parser.add_argument("--prompt-version", choices=sorted(PROMPTS), default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--source", choices=["official", "engineering"], default="official",
                        help="corpus whose claims get a query (engineering = larger, superset)")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--limit", type=int, default=None, help="generate at most N new queries")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--sample", type=int, default=0, help="also export N rows for human review")
    parser.add_argument("--sample-out", type=Path, default=Path("review_queries_sample.xlsx"))
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--no-generate", action="store_true", help="only export the sample")
    args = parser.parse_args(argv)

    items = candidate_items(args.corpus_dir, args.source)
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
        print(f"prompt {args.prompt_version}")
        entries = generate(items, args.output, llm, args.model, version,
                           args.limit, args.checkpoint_every, args.prompt_version)

    n_leak = sum(1 for e in entries if e["leaks_truth_tokens"] or e["leaks_poison_tokens"])
    n_num = sum(
        1 for e in entries
        if any(t[0].isdigit() for t in e["leaks_truth_tokens"] + e["leaks_poison_tokens"])
    )
    print(f"{len(entries)} queries in {args.output}; {n_leak} repeat a changed word, "
          f"{n_num} repeat a changed number or year (only these are excluded from experiments)")
    print(f"sha256: {hashlib.sha256(args.output.read_bytes()).hexdigest()}")

    if args.sample:
        rows = stratified_sample(items, entries, args.sample, args.sample_seed)
        write_review_sheet(rows, args.sample_out)
        print(f"Review sample of {len(rows)} rows written to {args.sample_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
