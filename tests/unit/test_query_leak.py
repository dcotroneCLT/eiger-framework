"""Tests for eiger.utils.query_leak."""

from eiger.utils.query_leak import leak_tokens, numeric_leak_tokens, tokens


def test_tokens_drop_stopwords_single_letters_and_keep_numbers() -> None:
    assert tokens("The rate rose to 3.5% in 2023") == {"rate", "rose", "3.5", "2023"}
    assert tokens("The U.S. economy grew 2 points") == {"economy", "grew", "2", "points"}


def test_leak_tokens_split_truth_and_poison() -> None:
    orig, mod = "Donated $50 million in 2022.", "Donated $55 million in 2021."
    assert leak_tokens("How much was donated?", orig, mod) == ([], [])
    assert leak_tokens("Was $50 million donated in 2021?", orig, mod) == (["50"], ["2021"])


def test_numeric_leak_ignores_words() -> None:
    orig = "The plan cut health insurance costs by 10 percent in 2020."
    mod = "The plan cut costs by 10 percent in 2021."
    assert numeric_leak_tokens("How did the plan affect health insurance costs?", orig, mod) == []
    assert numeric_leak_tokens("What did the plan do in 2020?", orig, mod) == ["2020"]
