"""Compute total years of experience from job date ranges (done in Python, not by the LLM)."""
from __future__ import annotations

import re
from datetime import date

from .schema import ResumeData

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_PRESENT = {"present", "current", "now", "ongoing", "today", "to date", "till date"}


def parse_date(s: str | None, *, is_end: bool, today: date) -> tuple[int, int] | None:
    """Return (year, month) or None. Year-only: start -> Jan, end -> Dec. 'Present' -> today."""
    if not s:
        return None
    s = s.strip().lower().rstrip(".")
    if s in _PRESENT:
        return today.year, today.month
    year_m = re.search(r"\b(19|20)\d{2}\b", s)
    if not year_m:
        return None
    year = int(year_m.group())
    month = None
    if m := re.search(r"\b([a-z]{3})[a-z]*\b", s):
        month = _MONTHS.get(m.group(1))
    if month is None and (m := re.fullmatch(r"(\d{4})-(\d{1,2})|(\d{1,2})[/-](\d{4})", s)):
        month = int(m.group(2) or m.group(3))
        if not 1 <= month <= 12:
            month = None
    if month is None:
        month = 12 if is_end else 1
    result = (year, month)
    return min(result, (today.year, today.month))  # no future end dates


def compute_years(data: ResumeData, today: date | None = None) -> ResumeData:
    """Fill total_years_experience (union of job intervals, so overlaps count once)."""
    today = today or date.today()
    intervals: list[tuple[int, int]] = []  # month indices, end exclusive
    for e in data.experience:
        label = e.title or e.company or "unnamed role"
        start = parse_date(e.start, is_end=False, today=today)
        end = parse_date(e.end, is_end=True, today=today)
        if start is None:
            data.warnings.append(f"Skipped '{label}': no usable start date.")
            continue
        if end is None:
            data.warnings.append(f"Skipped '{label}': no usable end date.")
            continue
        s, t = start[0] * 12 + start[1] - 1, end[0] * 12 + end[1]
        if t <= s:
            data.warnings.append(f"Skipped '{label}': end date is before start date.")
            continue
        intervals.append((s, t))

    if not intervals:
        data.total_years_experience = None
        return data

    intervals.sort()
    total, (cur_s, cur_e) = 0, intervals[0]
    for s, t in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, t)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, t
    total += cur_e - cur_s
    data.total_years_experience = round(total / 12, 1)
    return data
