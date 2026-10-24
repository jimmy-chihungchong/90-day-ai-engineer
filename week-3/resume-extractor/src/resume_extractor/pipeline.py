"""Glue: uploaded file -> text -> ResumeData, returning a UI-friendly result (never raises)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import Settings
from .extractor import ExtractionError, extract_resume
from .llm import LLMClient, LLMError, LLMTimeout
from .parsers import ParseError, parse_resume
from .schema import ResumeData


@dataclass
class RunResult:
    filename: str
    text: str = ""
    data: ResumeData | None = None
    error: str | None = None
    timed_out: bool = False
    suggestions: list[str] = field(default_factory=list)
    raw_reply: str = ""
    seconds: float = 0.0
    model: str = ""


def run_pipeline(filename: str, content: bytes, settings: Settings, client=None) -> RunResult:
    res = RunResult(filename, model=settings.model)
    start = time.perf_counter()
    try:
        parsed = parse_resume(filename, content)
        res.text = parsed.text
        client = client or LLMClient(settings)
        res.data = extract_resume(parsed.text, client, max_chars=settings.max_resume_chars)
        res.data.warnings = parsed.warnings + res.data.warnings
    except ParseError as e:
        res.error = str(e)
    except LLMTimeout as e:
        res.error, res.timed_out, res.suggestions = str(e), True, e.suggestions
    except LLMError as e:
        res.error, res.suggestions = str(e), e.suggestions
    except ExtractionError as e:
        res.error, res.raw_reply = str(e), e.raw_reply
    res.seconds = time.perf_counter() - start
    return res
