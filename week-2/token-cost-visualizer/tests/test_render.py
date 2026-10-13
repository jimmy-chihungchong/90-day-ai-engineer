from tokviz.render import chips_markdown, visible
from tokviz.tokenizer import tokenize


def test_visible_whitespace_and_escape():
    assert visible(" a\nb") == "·a↵b"
    assert visible("[x](y)") == r"\[x\]\(y\)"


def test_chips_one_per_token_and_cap():
    toks = tokenize("hello world")
    assert chips_markdown(toks).count("-background[") == len(toks)
    big = tokenize("a " * 3000)
    assert "more tokens not shown" in chips_markdown(big)
