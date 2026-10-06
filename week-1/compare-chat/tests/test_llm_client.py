import pytest

from chatbot.config import get_backend
from chatbot.llm_client import LLMError, LLMTimeout, chat, split_thinking, stream_chat


def test_split_thinking():
    t, a = split_thinking("<think>hmm</think>Answer")
    assert t == "hmm" and a == "Answer"
    assert split_thinking("plain") == ("", "plain")


def test_unreachable_server():
    b = get_backend("local")
    b = type(b)("local", "http://localhost:9/v1", "x", b.models)
    with pytest.raises((LLMError, LLMTimeout)):
        list(stream_chat(b, b.models[0], [{"role": "user", "content": "hi"}], timeout=2))


@pytest.mark.live
def test_live_llama():
    b = get_backend("local")
    r = chat(b, b.models[1], [{"role": "user", "content": "Say hi in 3 words."}], timeout=60)
    assert r.text.strip()


@pytest.mark.live
def test_live_timeout_suggests_other():
    b = get_backend("local")
    with pytest.raises(LLMTimeout) as e:
        chat(b, b.models[0], [{"role": "user", "content": "Prove sqrt(2) is irrational in detail."}], timeout=0.001)
    assert e.value.suggestion == b.models[1]


def test_boxed_stripped():
    assert split_thinking("<think>x</think>\\boxed{apple}")[1] == "apple"
