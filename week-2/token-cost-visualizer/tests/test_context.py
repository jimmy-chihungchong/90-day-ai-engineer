import pytest

from tokviz.context import split_window, text_of
from tokviz.tokenizer import tokenize

TOKENS = tokenize("one two three four five six seven eight nine ten")   # 10 tokens


def test_everything_fits():
    s = split_window(TOKENS, window=20)
    assert s.kept == TOKENS and s.n_dropped == 0 and s.budget == 20


def test_output_reserve_shrinks_input_budget():
    s = split_window(TOKENS, window=12, reserve_output=4)
    assert s.budget == 8 and len(s.kept) == 8 and s.n_dropped == 2


def test_keep_recent_drops_oldest():
    s = split_window(TOKENS, window=4, keep="recent")
    assert text_of(s.kept).strip() == "seven eight nine ten"
    assert text_of(s.dropped_start).startswith("one") and s.dropped_end == []


def test_keep_start_drops_newest():
    s = split_window(TOKENS, window=4, keep="start")
    assert text_of(s.kept).strip() == "one two three four"
    assert text_of(s.dropped_end).strip().endswith("ten") and s.dropped_start == []


def test_parts_always_rebuild_original():
    for keep in ("recent", "start"):
        s = split_window(TOKENS, window=3, reserve_output=1, keep=keep)
        assert s.dropped_start + s.kept + s.dropped_end == TOKENS


def test_reserve_bigger_than_window_keeps_nothing():
    s = split_window(TOKENS, window=5, reserve_output=9)
    assert s.budget == 0 and s.kept == [] and s.n_dropped == 10


def test_empty_and_bad_keep():
    assert split_window([], 10).kept == []
    with pytest.raises(ValueError):
        split_window(TOKENS, 5, keep="middle")
