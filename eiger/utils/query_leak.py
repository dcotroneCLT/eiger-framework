"""
Detect retrieval queries that favour one of two competing documents.

A query compared against a ground-truth claim and its manipulated variant is
neutral only if it repeats none of the tokens that differ between the two
texts. If it repeats a token present only in the original (e.g. the true
year), it favours the ground truth; if it repeats a token present only in the
manipulated text, it favours the poison.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?", re.IGNORECASE)
_STOPWORDS = frozenset(
    "a an the of in on at to for from by with and or but not no is are was were be been "
    "it its this that these those as than then there their they he she his her we you i "
    "has have had do does did will would can could may might should about into over".split()
)


def tokens(text: str) -> set[str]:
    """Lower-cased word/number tokens, without common stopwords."""
    return {t.lower() for t in _TOKEN_RE.findall(text)} - _STOPWORDS


def leak_tokens(query: str, original: str, modified: str) -> tuple[list[str], list[str]]:
    """Return (truth-only tokens, poison-only tokens) that appear in the query."""
    q, o, m = tokens(query), tokens(original), tokens(modified)
    return sorted(q & (o - m)), sorted(q & (m - o))
