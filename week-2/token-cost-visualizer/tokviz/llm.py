"""OpenAI-compatible client (LM Studio / NIM): sampling params, logprobs, timeout + fallback suggestion."""
import time
from dataclasses import dataclass, field

import openai
from openai import OpenAI

from .config import Backend

# When a model is slow, suggest the first of these that is available (fast/small first).
FALLBACK_PREFERENCE = ["llama-3.2-3b", "qwen3-14b", "gemma", "gpt-oss-20b", "muse-glimmer"]


class LLMTimeout(Exception):
    """The model produced nothing within the timeout window."""

    def __init__(self, model: str, suggestion: str | None):
        self.model, self.suggestion = model, suggestion
        hint = f" Try {suggestion} instead." if suggestion else ""
        super().__init__(f"{model} gave no response in time.{hint}")


class LLMError(Exception):
    pass


@dataclass
class TokenLP:
    token: str
    logprob: float
    top: list[tuple[str, float]] = field(default_factory=list)  # alternatives, best first


@dataclass
class Completion:
    model: str
    text: str
    reasoning: str
    tokens: list[TokenLP]       # empty unless logprobs=True and supported
    prompt_tokens: int
    completion_tokens: int
    latency: float


def make_client(backend: Backend, timeout: float) -> OpenAI:
    return OpenAI(base_url=backend.base_url, api_key=backend.api_key or "none",
                  timeout=timeout, max_retries=0)


def list_models(backend: Backend, timeout: float = 5.0, client: OpenAI | None = None) -> list[str]:
    client = client or make_client(backend, timeout)
    try:
        return [m.id for m in client.models.list().data]
    except openai.APIConnectionError as e:
        raise LLMError(f"Cannot reach {backend.base_url} ({backend.name}). Is the server running? {e}")
    except openai.APIStatusError as e:
        raise LLMError(f"{backend.name} returned {e.status_code}: {e.message}")


def suggest_alternative(current: str, available: list[str]) -> str | None:
    """Pick another chat model to try (skips embedding models and the current one)."""
    chat = [m for m in available if m != current and "embed" not in m.lower()]
    for pref in FALLBACK_PREFERENCE:
        for m in chat:
            if pref in m.lower():
                return m
    return chat[0] if chat else None


def complete(backend: Backend, model: str, prompt: str, *, temperature: float = 0.7,
             top_p: float = 1.0, max_tokens: int = 300, logprobs: bool = False,
             top_logprobs: int = 5, system: str | None = None, timeout: float = 45.0,
             client: OpenAI | None = None, available: list[str] | None = None) -> Completion:
    """One non-streaming completion. Raises LLMTimeout (with a suggested model) or LLMError."""
    client = client or make_client(backend, timeout)
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    kwargs = dict(model=model, messages=messages, temperature=temperature,
                  top_p=top_p, max_tokens=max_tokens)
    if logprobs:
        kwargs.update(logprobs=True, top_logprobs=top_logprobs)
    t0 = time.time()
    try:
        r = client.chat.completions.create(**kwargs)
    except openai.APITimeoutError:
        raise LLMTimeout(model, suggest_alternative(model, available or backend_models(backend)))
    except openai.APIConnectionError as e:
        raise LLMError(f"Cannot reach {backend.base_url} ({backend.name}). Is the server running? {e}")
    except openai.APIStatusError as e:
        raise LLMError(f"{backend.name} returned {e.status_code}: {e.message}")

    ch = r.choices[0]
    lp = ch.logprobs.content if ch.logprobs and ch.logprobs.content else []
    tokens = [TokenLP(x.token, x.logprob, [(a.token, a.logprob) for a in (x.top_logprobs or [])])
              for x in lp]
    reasoning = getattr(ch.message, "reasoning_content", None) or getattr(ch.message, "reasoning", None) or ""
    usage = r.usage
    return Completion(model, ch.message.content or "", reasoning, tokens,
                      usage.prompt_tokens if usage else 0,
                      usage.completion_tokens if usage else 0, time.time() - t0)


def backend_models(backend: Backend) -> list[str]:
    """Best-effort model list for fallback suggestions; never raises."""
    try:
        return list_models(backend, timeout=3.0)
    except Exception:
        return []
