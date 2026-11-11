"""
Detect retrieval queries that favour one of two competing documents.

A query compared against a ground-truth claim and its manipulated variant is
neutral only if it repeats none of the tokens that differ between the two
texts. If it repeats a token present only in the original (e.g. the true
year), it favours the ground truth; if it repeats a token present only in the
manipulated text, it favours the poison.

Exclusion policy (agreed with the corpus author, 2026-11-06): only a
repeated NUMBER or YEAR excludes a query (``numeric_leak_tokens``). For
context omission (M06), a natural question about the subject usually names
the omitted part; that is the realistic journalist query, and excluding it
would keep only claims where the omission is irrelevant to the question,
making the attack look stronger than it is. Word-level leaks are still
reported by ``leak_tokens`` for analysis.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"\d[\d.,]*")
_STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or but not no is are was were be been "
    "it its this that these those as than then there their they he she his her we you i "
    "has have had do does did will would can could may might should about into over".split()
)


def tokens(text: str) -> set[str]:
    """
    Lower-cased word/number tokens, without common stopwords and without
    single letters (so "U.S." does not produce the tokens "u" and "s").
    """
    found = {t.lower() for t in _TOKEN_RE.findall(text)}
    return {t for t in found if len(t) > 1 or t.isdigit()} - _STOPWORDS


def leak_tokens(query: str, original: str, modified: str) -> tuple[list[str], list[str]]:
    """Return (truth-only tokens, poison-only tokens) that appear in the query."""
    q, o, m = tokens(query), tokens(original), tokens(modified)
    return sorted(q & (o - m)), sorted(q & (m - o))


def numeric_leak_tokens(query: str, original: str, modified: str) -> list[str]:
    """Numbers or years that differ between the two texts and appear in the query."""
    truth, poison = leak_tokens(query, original, modified)
    return sorted(t for t in truth + poison if _NUMERIC_RE.fullmatch(t))
