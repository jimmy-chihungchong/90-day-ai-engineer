"""NIM backend: config, embedding hints, sidebar switch, and live checks (skipped without NIM_API_KEY)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from tests.test_embeddings import FakeClient
from tokviz.config import (NIM_DEFAULT_CHAT, NIM_DEFAULT_EMBED, NIM_VERIFIED_CHAT, Backend,
                           backend_options, get_backend)
from tokviz.embeddings import cosine_matrix, embed
from tokviz.llm import complete

APP = Path(__file__).resolve().parent.parent / "app.py"
HAS_KEY = bool(get_backend("nim").api_key)
live_nim = pytest.mark.skipif(not HAS_KEY, reason="NIM_API_KEY not set")


def test_nim_defaults(monkeypatch):
    for k in ("NIM_CHAT_MODEL", "NIM_EMBED_MODEL", "NIM_CHAT_MODELS"):
        monkeypatch.delenv(k, raising=False)
    b = get_backend("nim")
    assert b.chat_model == NIM_DEFAULT_CHAT and b.embed_model == NIM_DEFAULT_EMBED
    assert b.chat_models == NIM_VERIFIED_CHAT and b.chat_models[0] == NIM_DEFAULT_CHAT


def test_nim_model_list_override(monkeypatch):
    monkeypatch.setenv("NIM_CHAT_MODELS", " a/one , b/two ,")
    assert get_backend("nim").chat_models == ("a/one", "b/two")


def test_local_backend_discovers_models_from_server():
    assert get_backend("local").chat_models == ()


def test_backend_options_hide_local_on_cloud(monkeypatch):
    monkeypatch.setenv("BACKEND", "nim")
    assert backend_options() == ["nim"]
    monkeypatch.setenv("BACKEND", "local")
    assert backend_options() == ["local", "nim"]


def test_embed_sends_input_type_only_on_nim():
    nim, local = Backend("nim", "http://x", "k", "c", "e"), Backend("local", "http://x", "k", "c", "e")
    for backend, kind, expected in [(nim, "passage", {"input_type": "passage"}),
                                    (nim, "query", {"input_type": "query"}), (local, "query", None)]:
        c = FakeClient(vectors=[[1.0, 0.0]])
        embed(backend, ["a"], client=c, kind=kind)
        assert c.kw.get("extra_body") == expected


def test_sidebar_defaults_and_nim_without_key_warns(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "")
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["backend"] = "nim"
    at.run()
    assert not at.exception
    assert [r.value for r in at.sidebar.radio] == ["nim"]
    assert any("NIM_API_KEY" in w.value for w in at.sidebar.warning)


# ---- live NIM checks (cost a few tiny requests) ----
@pytest.fixture
def code_defaults(monkeypatch):
    """Ignore model overrides in the developer's .env so the tests exercise the shipped defaults."""
    for k in ("NIM_CHAT_MODEL", "NIM_EMBED_MODEL", "NIM_CHAT_MODELS"):
        monkeypatch.delenv(k, raising=False)


@live_nim
def test_live_nim_embeddings_separate_meaning(code_defaults):
    b = get_backend("nim")
    v = embed(b, ["The cat sat on the sofa.", "A kitten is sleeping on the couch.",
                  "The stock market fell sharply today."], timeout=60)
    m = cosine_matrix(v)
    assert m[0, 1] > m[0, 2] + 0.1


@live_nim
def test_live_nim_default_chat_returns_text_and_logprobs(code_defaults):
    b = get_backend("nim")
    r = complete(b, b.chat_model, "Capital of Australia? One word.", temperature=0, max_tokens=40,
                 logprobs=True, top_logprobs=3, timeout=60, available=list(b.chat_models))
    assert "Canberra" in r.text and r.tokens and not r.reasoning


@live_nim
def test_live_nim_app_runs_all_llm_tabs(code_defaults):
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.session_state["backend"] = "nim"
    at.run()
    for label in ("Generate", "Ask", "Embed and map"):
        next(b for b in at.button if b.label == label).click().run()
        assert not at.exception and not at.error, label
    assert any(m.label == "Average confidence" for m in at.metric)
