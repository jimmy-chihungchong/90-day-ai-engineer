import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer as HTTPServer
import socket

import pytest

from resume_extractor.config import Settings, get_settings
from resume_extractor.llm import LLMClient, LLMError, LLMTimeout, LLMUnavailable, extract_json


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"data": [{"id": "slow-model"}, {"id": "fast-model"}, {"id": "nomic-embed-x"}]})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers["Content-Length"]))
        time.sleep(self.server.delay)
        body = json.loads(raw)
        self.server.seen_max_tokens.append(body["max_tokens"])
        if self.server.mode == "length_once" and len(self.server.seen_max_tokens) == 1:
            return self._send({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})
        if self.server.mode == "always_length":
            return self._send({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})
        if self.server.mode == "context":
            msg = json.dumps({"error": {"message": "Trying to keep 9000 tokens but context length is 8192"}}).encode()
            self.send_response(400); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg))); self.end_headers()
            return self.wfile.write(msg)
        self._send({"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]})


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    srv.delay, srv.mode, srv.seen_max_tokens = 0, "ok", []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def settings_for(srv, timeout=5.0):
    return Settings("lmstudio", f"http://127.0.0.1:{srv.server_port}/v1", "slow-model", "x", timeout)


def test_complete_ok(server):
    assert LLMClient(settings_for(server)).complete("s", "u") == '{"ok": true}'


def test_timeout_suggests_other_chat_models(server):
    server.delay = 2
    with pytest.raises(LLMTimeout) as ei:
        LLMClient(settings_for(server, timeout=0.5)).complete("s", "u")
    assert ei.value.suggestions == ["fast-model"]  # not itself, not the embedding model


def test_unreachable():
    s = Settings("lmstudio", f"http://127.0.0.1:{_free_port()}/v1", "m", "x", 2)
    with pytest.raises(LLMUnavailable):
        LLMClient(s).complete("s", "u")


@pytest.mark.parametrize("raw", ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! Here:\n{"a": 1}\nDone.'])
def test_extract_json(raw):
    assert extract_json(raw) == {"a": 1}


def test_extract_json_failure():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_settings(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "nim"); monkeypatch.delenv("NIM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="NIM_API_KEY"):
        get_settings({"NIM_API_KEY": ""})
    assert get_settings({"NIM_API_KEY": "k"}).api_key == "k"
    monkeypatch.setenv("LLM_BACKEND", "bogus")
    with pytest.raises(ValueError):
        get_settings()


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def test_default_budget_is_8192(server):
    LLMClient(settings_for(server)).complete("s", "u")
    assert server.seen_max_tokens == [8192]


def test_retries_once_with_double_budget_when_reasoning_exhausts_tokens(server):
    server.mode = "length_once"
    assert LLMClient(settings_for(server)).complete("s", "u") == '{"ok": true}'
    assert server.seen_max_tokens == [8192, 16384]


def test_gives_clear_error_when_still_empty(server):
    server.mode = "always_length"
    with pytest.raises(LLMError, match="empty answer"):
        LLMClient(settings_for(server)).complete("s", "u")
    assert len(server.seen_max_tokens) == 2  # one retry only


def test_context_overflow_message(server):
    server.mode = "context"
    with pytest.raises(LLMError, match="context window"):
        LLMClient(settings_for(server)).complete("s", "u")
