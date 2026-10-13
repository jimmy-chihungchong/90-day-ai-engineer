import httpx
import openai
import pytest

from tokviz.config import get_backend
from tokviz.llm import LLMError, LLMTimeout, complete, list_models, suggest_alternative

REQ = httpx.Request("POST", "http://x/v1/chat/completions")
B = get_backend("local")


class FakeClient:
    """Minimal stand-in for OpenAI(): .chat.completions.create raises or returns."""
    def __init__(self, exc=None, result=None):
        self.chat = self; self.completions = self
        self._exc, self._result = exc, result

    def create(self, **kw):
        self.kw = kw
        if self._exc:
            raise self._exc
        return self._result


def test_suggest_skips_current_and_embeddings():
    avail = ["text-embedding-nomic-embed-text-v1.5", "openai/gpt-oss-20b", "llama-3.2-3b-instruct"]
    assert suggest_alternative("openai/gpt-oss-20b", avail) == "llama-3.2-3b-instruct"
    assert suggest_alternative("llama-3.2-3b-instruct", avail) == "openai/gpt-oss-20b"
    assert suggest_alternative("a", ["text-embedding-x", "a"]) is None


def test_timeout_raises_with_suggestion():
    c = FakeClient(exc=openai.APITimeoutError(request=REQ))
    with pytest.raises(LLMTimeout) as e:
        complete(B, "openai/gpt-oss-20b", "hi", client=c, available=["openai/gpt-oss-20b", "qwen/qwen3-14b"])
    assert e.value.suggestion == "qwen/qwen3-14b" and "qwen3-14b" in str(e.value)


def test_connection_error_is_friendly():
    c = FakeClient(exc=openai.APIConnectionError(request=REQ))
    with pytest.raises(LLMError, match="Is the server running"):
        complete(B, "m", "hi", client=c)


def test_status_error_is_friendly():
    resp = httpx.Response(404, request=REQ, json={"error": {"message": "model not found"}})
    c = FakeClient(exc=openai.NotFoundError("model not found", response=resp, body=None))
    with pytest.raises(LLMError, match="404"):
        complete(B, "m", "hi", client=c)


# ---- live tests against LM Studio (skipped if it isn't running) ----
def _live():
    try:
        return list_models(B, timeout=2)
    except LLMError:
        return None


live = pytest.mark.skipif(_live() is None, reason="LM Studio not running")


@live
def test_live_complete_with_logprobs():
    r = complete(B, "llama-3.2-3b-instruct", "Capital of Australia? One word.",
                 temperature=0, max_tokens=50, logprobs=True, top_logprobs=3, timeout=90)
    assert "Canberra" in r.text
    assert r.tokens and r.tokens[0].top and r.completion_tokens > 0 and r.latency > 0


@live
def test_live_timeout_suggests_other_model():
    with pytest.raises(LLMTimeout) as e:
        complete(B, "openai/gpt-oss-20b", "Write a 500 word essay on the sea.",
                 max_tokens=1500, timeout=0.5)
    assert e.value.suggestion and e.value.suggestion != "openai/gpt-oss-20b"
