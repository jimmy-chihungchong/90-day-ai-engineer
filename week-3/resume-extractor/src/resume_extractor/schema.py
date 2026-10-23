"""Module 2: Pydantic models defining the output JSON."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


def _none_to_list(v: Any) -> Any:
    # LLMs often emit null / a bare string where a list is expected.
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return v


def _to_str(v: Any) -> Any:
    # e.g. start: 2019 (int) instead of "2019"
    return str(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


class Experience(BaseModel):
    company: str | None = None
    title: str | None = None
    start: str | None = Field(None, description="YYYY-MM or YYYY")
    end: str | None = Field(None, description="YYYY-MM, YYYY, or 'Present'")
    description: str | None = None

    _dates = field_validator("start", "end", mode="before")(_to_str)


class Education(BaseModel):
    institution: str | None = None
    degree: str | None = None
    start: str | None = None
    end: str | None = None

    _dates = field_validator("start", "end", mode="before")(_to_str)


class ResumeData(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    links: list[str] = []
    summary: str | None = None
    skills: list[str] = []
    experience: list[Experience] = []
    education: list[Education] = []
    # Computed in Python (experience.py), never trusted from the LLM.
    total_years_experience: float | None = None
    warnings: list[str] = []

    _lists = field_validator("links", "skills", "experience", "education", "warnings",
                             mode="before")(_none_to_list)

    @field_validator("skills")
    @classmethod
    def _dedupe_skills(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        out = []
        for s in v:
            s = str(s).strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                out.append(s)
        return out
