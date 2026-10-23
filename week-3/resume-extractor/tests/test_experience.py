from datetime import date

import pytest

from resume_extractor.experience import compute_years, parse_date
from resume_extractor.schema import Experience, ResumeData

TODAY = date(2026, 6, 15)


def years(*jobs):
    d = ResumeData(experience=[Experience(start=s, end=e) for s, e in jobs])
    return compute_years(d, TODAY)


@pytest.mark.parametrize("s,is_end,expected", [
    ("2019-03", False, (2019, 3)),
    ("03/2019", False, (2019, 3)),
    ("Mar 2019", False, (2019, 3)),
    ("September 2020", True, (2020, 9)),
    ("2019", False, (2019, 1)),
    ("2019", True, (2019, 12)),
    ("Present", True, (2026, 6)),
    ("current", True, (2026, 6)),
    ("2030", True, (2026, 6)),      # future end clamped
    ("sometime", True, None),
    (None, False, None),
])
def test_parse_date(s, is_end, expected):
    assert parse_date(s, is_end=is_end, today=TODAY) == expected


def test_simple_range():
    assert years(("2020-01", "2022-12")).total_years_experience == 3.0


def test_present():
    assert years(("2024-06", "Present")).total_years_experience == 2.1


def test_overlap_counted_once():
    assert years(("2018-01", "2019-12"), ("2019-01", "2020-12")).total_years_experience == 3.0


def test_gap_not_counted():
    assert years(("2018-01", "2018-12"), ("2020-01", "2020-12")).total_years_experience == 2.0


def test_bad_rows_warn_and_skip():
    d = years((None, "2020"), ("2020", "2018"), ("2019", None), ("2021-01", "2021-12"))
    assert d.total_years_experience == 1.0
    assert len(d.warnings) == 3


def test_no_experience_is_none():
    assert compute_years(ResumeData(), TODAY).total_years_experience is None


def test_schema_coerces_llm_sloppiness():
    d = ResumeData.model_validate({"skills": ["Python", "python", " SQL ", ""], "links": None,
                                   "education": None, "extra_field": 1})
    assert d.skills == ["Python", "SQL"] and d.links == [] and d.education == []
