"""Module 3: OpenAI-compatible client wrapper with timeout + fallback hint."""
from __future__ import annotations

import json
import re

import openai
from openai import OpenAI

from .config import NIM_FALLBACKS, Settings


class LLMError(Exception):
    """Base error; `suggestions` lists other models the user could try."""

    def __init__(self, message: str, suggestions: list[str] | None = None):
        super().__init__(message)
        self.suggestions = suggestions or []


class LLMTimeout(LLMError):
    pass


class LLMUnavailable(LLMError):
    pass


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        # max_retries=0: we want the timeout to surface to the user, not be hidden by retries.
        self._client = OpenAI(base_url=settings.base_url, api_key=settings.api_key,
                              timeout=openai.Timeout(settings.timeout, connect=10.0),
                              max_retries=0)

    def list_models(self) -> list[str]:
        try:
            ids = [m.id for m in self._client.models.list().data]
        except openai.OpenAIError:
            return []
        return [i for i in ids if "embed" not in i.lower()]

    def suggest_alternatives(self, current: str) -> list[str]:
        if self.settings.backend == "nim":
            pool = NIM_FALLBACKS
        else:
            pool = self.list_models()
        return [m for m in pool if m != current][:3]

    def _create(self, model, messages, max_tokens):
        return self._client.chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=max_tokens)

    @staticmethod
    def _ran_out_of_tokens(resp) -> bool:
        choice = resp.choices[0]
        return choice.finish_reason == "length" and not (choice.message.content or "").strip()

    def complete(self, system: str, user: str, model: str | None = None,
                 max_tokens: int | None = None) -> str:
        """Return the assistant's text. Raises LLMTimeout / LLMUnavailable / LLMError."""
        model = model or self.settings.model
        if self.settings.backend == "lmstudio":
            # LM Studio + gpt-oss-20b returns a server-side "peg-native format" error whenever a
            # system message is present, so fold it into the user turn.
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
        else:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
        budget = max_tokens or self.settings.max_tokens
        try:
            resp = self._create(model, messages, budget)
            if self._ran_out_of_tokens(resp):
                # Reasoning models can burn the whole budget thinking; retry once with double.
                resp = self._create(model, messages, budget * 2)
        except openai.APITimeoutError as e:
            raise LLMTimeout(
                f"'{model}' did not answer within {self.settings.timeout:.0f}s.",
                self.suggest_alternatives(model),
            ) from e
        except openai.APIConnectionError as e:
            raise LLMUnavailable(
                f"Cannot reach {self.settings.base_url}. Is the server running?"
            ) from e
        except openai.AuthenticationError as e:
            raise LLMError("The API key was rejected.") from e
        except openai.NotFoundError as e:
            raise LLMError(f"Model '{model}' was not found on this server.",
                           self.suggest_alternatives(model)) from e
        except openai.BadRequestError as e:
            if "context" in str(e).lower():
                raise LLMError(
                    f"The resume does not fit in the model's context window ({e}). Use a shorter "
                    "resume, lower MAX_RESUME_CHARS, or load the model with a larger context "
                    "length (LM Studio: model settings > Context Length).") from e
            raise LLMError(f"LLM request failed: {e}") from e
        except openai.OpenAIError as e:
            raise LLMError(f"LLM request failed: {e}") from e

        text = resp.choices[0].message.content or ""
        if not text.strip():
            raise LLMError(f"'{model}' returned an empty answer "
                           f"(finish_reason={resp.choices[0].finish_reason}); "
                           "it may have run out of tokens while reasoning.",
                           self.suggest_alternatives(model))
        return text


def extract_json(text: str) -> dict:
    """Pull a JSON object out of an LLM reply (handles ``` fences and chatter)."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in the model reply.")
    return json.loads(text[start:end + 1])
