"""Module 4: text -> prompt -> LLM -> validated ResumeData."""
from __future__ import annotations

from datetime import date
from typing import Protocol

from pydantic import ValidationError

from .experience import compute_years
from .llm import extract_json
from .schema import ResumeData

MAX_CHARS = 24_000  # ~6k tokens; longer resumes are truncated with a warning

SYSTEM_PROMPT = """You extract structured data from resume text. Reply with ONE JSON object and nothing else.

Schema (use null for unknown scalars, [] for unknown lists):
{
  "name": string|null,
  "email": string|null,
  "phone": string|null,
  "location": string|null,
  "links": [string],
  "summary": string|null,
  "skills": [string],
  "experience": [{"company": string|null, "title": string|null,
                  "start": string|null, "end": string|null, "description": string|null}],
  "education": [{"institution": string|null, "degree": string|null,
                 "start": string|null, "end": string|null}]
}

Rules:
- Use only information present in the text. Never invent or guess values.
- Dates: "YYYY-MM" when the month is known, otherwise "YYYY". Use "Present" for a current role.
- "skills": one flat list of individual skills/tools/languages, gathered from every section.
- "experience": paid roles, internships and freelance work only, most recent first.
  Put a 1-2 sentence summary of the role in "description".
- "links": URLs such as LinkedIn, GitHub, portfolio.
- The text may be messy: columns interleaved, odd bullets, tables rendered as "a | b"."""


class _LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


def extract_resume(text: str, llm: _LLM, today: date | None = None) -> ResumeData:
    """Run the full text -> ResumeData pipeline. Raises LLMError subclasses from the client."""
    warnings: list[str] = []
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        warnings.append(f"Resume text was truncated to {MAX_CHARS} characters.")

    user = f"Resume text:\n\"\"\"\n{text}\n\"\"\""
    reply = llm.complete(SYSTEM_PROMPT, user)
    try:
        data = _parse(reply)
    except (ValueError, ValidationError) as first_err:
        # One repair attempt: show the model its own output and what was wrong with it.
        repair = (f"{user}\n\nYour previous reply was invalid: {_short(first_err)}\n"
                  f"Previous reply:\n{reply[:4000]}\n\nReply again with only the corrected JSON object.")
        try:
            data = _parse(llm.complete(SYSTEM_PROMPT, repair))
        except (ValueError, ValidationError) as second_err:
            raise ExtractionError(
                f"The model did not return valid JSON after a retry: {_short(second_err)}",
                raw_reply=reply) from second_err
        warnings.append("First model reply was invalid; the retry succeeded.")

    data.warnings = warnings + data.warnings
    return compute_years(data, today)


def _parse(reply: str) -> ResumeData:
    return ResumeData.model_validate(extract_json(reply))


def _short(err: Exception) -> str:
    return " ".join(str(err).split())[:300]


class ExtractionError(Exception):
    def __init__(self, message: str, raw_reply: str = ""):
        super().__init__(message)
        self.raw_reply = raw_reply
