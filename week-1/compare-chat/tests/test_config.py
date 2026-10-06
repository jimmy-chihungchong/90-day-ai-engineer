from chatbot.config import get_backend, get_timeout


def test_local_default(monkeypatch):
    monkeypatch.delenv("BACKEND", raising=False)
    b = get_backend()
    assert b.name == "local" and b.base_url.endswith(":1234/v1") and len(b.models) == 2


def test_nim(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "k")
    b = get_backend("nim")
    assert b.name == "nim" and b.api_key == "k"


def test_timeout(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "10")
    assert get_timeout() == 10.0
