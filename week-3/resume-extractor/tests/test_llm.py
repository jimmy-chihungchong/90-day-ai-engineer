import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer as HTTPServer
import socket

import pytest

from resume_extractor.config import Settings, get_settings
from resume_extractor.llm import LLMClient, LLMTimeout, LLMUnavailable, extract_json


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
        self.rfile.read(int(self.headers["Content-Length"]))
        if "slow" in self.path or True:
            time.sleep(self.server.delay)
        self._send({"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]})


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    srv.delay = 0
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
