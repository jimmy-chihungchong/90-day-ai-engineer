"""Module 2: OpenAI-compatible client (LM Studio / NIM), streaming, timeout."""
import re
import time
from dataclasses import dataclass

import openai
from openai import OpenAI

from .config import Backend


class LLMTimeout(Exception):
    """No tokens arrived within the timeout window."""

    def __init__(self, model, suggestion):
        self.model, self.suggestion = model, suggestion
        super().__init__(f"{model} gave no response in time. Try {suggestion} instead.")


class LLMError(Exception):
    pass


@dataclass
class Result:
    model: str
    text: str
    latency: float


def split_thinking(text):
    """Separate <think>...</think> reasoning from the final answer."""
    m = re.search(r"<think>(.*?)(</think>|$)", text, re.S)
    think, answer = ("", text) if not m else (
        m.group(1).strip(), (text[:m.start()] + text[m.end():]).strip())
    # phi-4-mini-reasoning wraps final answers in \boxed{...}
    answer = re.sub(r"\\boxed\{(.*?)\}", r"\1", answer, flags=re.S)
    return think, answer


def make_client(backend: Backend, timeout: float) -> OpenAI:
    # timeout applies per read, so it fires when the stream goes silent
    return OpenAI(base_url=backend.base_url, api_key=backend.api_key or "none",
                  timeout=timeout, max_retries=0)


def stream_chat(backend, model, messages, timeout=45.0, client=None):
    """Yield text chunks. Raises LLMTimeout / LLMError."""
    client = client or make_client(backend, timeout)
    other = next((m for m in backend.models if m != model), model)
    try:
        stream = client.chat.completions.create(model=model, messages=messages, stream=True)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except openai.APITimeoutError:
        raise LLMTimeout(model, other)
    except openai.APIConnectionError as e:
        raise LLMError(f"Cannot reach {backend.base_url} ({backend.name}). Is the server running? {e}")
    except openai.APIStatusError as e:
        raise LLMError(f"{backend.name} returned {e.status_code}: {e.message}")


def chat(backend, model, messages, timeout=45.0, client=None) -> Result:
    """Non-streaming convenience wrapper (used by compare)."""
    t0 = time.time()
    text = "".join(stream_chat(backend, model, messages, timeout, client))
    return Result(model, text, time.time() - t0)
