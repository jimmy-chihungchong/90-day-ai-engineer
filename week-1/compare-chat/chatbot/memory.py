"""Module 3: conversation memory + JSON export/import.

One Conversation holds a separate message history per model key, so compared
models never see each other's answers.
"""
import json
from datetime import datetime, timezone

SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."
MAX_CHARS = 12000  # crude context budget for small models


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Conversation:
    def __init__(self):
        self.histories = {}  # model -> list of {role, content, ts, latency?}

    def add(self, model, role, content, latency=None):
        msg = {"role": role, "content": content, "ts": _now()}
        if latency is not None:
            msg["latency"] = round(latency, 2)
        self.histories.setdefault(model, []).append(msg)

    def get(self, model):
        return self.histories.get(model, [])

    def for_llm(self, model):
        """System prompt + newest messages that fit the budget; role/content only."""
        kept, total = [], 0
        for m in reversed(self.get(model)):
            total += len(m["content"])
            if total > MAX_CHARS and kept:
                break
            kept.append({"role": m["role"], "content": m["content"]})
        return [{"role": "system", "content": SYSTEM_PROMPT}] + kept[::-1]

    def clear(self):
        self.histories = {}

    def to_json(self):
        return json.dumps({"version": 1, "exported_at": _now(), "histories": self.histories},
                          indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text):
        data = json.loads(text)
        hist = data.get("histories")
        if not isinstance(hist, dict):
            raise ValueError("Not a valid memory file (missing 'histories').")
        for msgs in hist.values():
            for m in msgs:
                if m.get("role") not in ("user", "assistant") or "content" not in m:
                    raise ValueError("Memory file contains an invalid message.")
        c = cls()
        c.histories = hist
        return c
