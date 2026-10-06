import pytest

from chatbot.memory import Conversation


def test_separate_histories_and_llm_format():
    c = Conversation()
    c.add("a", "user", "my name is Jim")
    c.add("a", "assistant", "hi Jim", latency=1.234)
    c.add("b", "user", "other")
    msgs = c.for_llm("a")
    assert msgs[0]["role"] == "system" and len(msgs) == 3
    assert set(msgs[1]) == {"role", "content"}
    assert len(c.for_llm("b")) == 2


def test_trim_keeps_newest():
    c = Conversation()
    for i in range(10):
        c.add("a", "user", "x" * 5000 + str(i))
    out = c.for_llm("a")
    assert out[-1]["content"].endswith("9") and len(out) < 11


def test_roundtrip():
    c = Conversation()
    c.add("a", "user", "héllo")
    c2 = Conversation.from_json(c.to_json())
    assert c2.get("a")[0]["content"] == "héllo"


def test_bad_import():
    with pytest.raises(ValueError):
        Conversation.from_json('{"foo": 1}')
    with pytest.raises(ValueError):
        Conversation.from_json('{"histories": {"a": [{"role": "bot", "content": "x"}]}}')
