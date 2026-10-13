import math

from tokviz.confidence import band, prob, summarize
from tokviz.llm import TokenLP
from tokviz.render import escape_md


def tok(text, p):
    return TokenLP(text, math.log(p))


def test_bands():
    assert band(0.99) == "sure" and band(0.9) == "sure"
    assert band(0.7) == "unsure" and band(0.5) == "unsure"
    assert band(0.1) == "shaky"


def test_prob_roundtrip():
    assert prob(tok("a", 0.25)) == __import__("pytest").approx(0.25)


def test_summarize():
    s = summarize([tok("The", 0.9), tok(" Tim", 0.1), tok(" Davis", 0.3), tok(".", 0.99)])
    assert s.lowest_token == " Tim" and s.lowest_p < 0.11
    assert s.n_shaky == 2 and s.n_tokens == 4
    assert 0.5 < s.mean < 0.6


def test_summarize_empty():
    assert summarize([]) is None


def test_escape_md_keeps_text_literal():
    assert escape_md("1,044.") == r"1,044\."
    assert escape_md("a_b*c") == r"a\_b\*c"


def test_escape_md_escapes_backslash_itself():
    assert escape_md("a\\b") == "a\\\\b"
