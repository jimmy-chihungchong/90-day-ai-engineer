from datetime import date

import pytest

from resume_extractor.extractor import MAX_CHARS, ExtractionError, extract_resume

GOOD = '''```json
{"name": "Jane Doe", "skills": ["Python", "python", "SQL"],
 "experience": [{"company": "Acme", "title": "Eng", "start": 2019, "end": "2021-12"},
                {"company": "Beta", "title": "Sr Eng", "start": "2022-01", "end": "Present"}],
 "total_years_experience": 99}
```'''


class FakeLLM:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def complete(self, system, user):
        self.calls.append(user)
        return self.replies.pop(0)


def test_happy_path_ignores_llm_years():
    d = extract_resume("x", FakeLLM(GOOD), today=date(2026, 1, 1))
    assert d.name == "Jane Doe" and d.skills == ["Python", "SQL"]
    assert d.experience[0].start == "2019"
    assert d.total_years_experience == 7.1   # 2019-01..2026-01 inclusive, contiguous, not the LLM's 99
    assert d.warnings == []


def test_retry_after_bad_json():
    llm = FakeLLM("sorry, I cannot", GOOD)
    d = extract_resume("x", llm, today=date(2026, 1, 1))
    assert len(llm.calls) == 2 and "invalid" in llm.calls[1]
    assert any("retry succeeded" in w for w in d.warnings)


def test_retry_after_schema_violation():
    llm = FakeLLM('{"experience": "oops-not-a-list-of-objects", "name": ["x"]}', GOOD)
    assert extract_resume("x", llm).name == "Jane Doe"


def test_gives_up_after_one_retry():
    with pytest.raises(ExtractionError) as ei:
        extract_resume("x", FakeLLM("nope", "still nope"))
    assert ei.value.raw_reply == "nope"


def test_long_text_keeps_head_and_tail():
    text = "HEAD-JOBS " + "x" * 5000 + " TAIL-SKILLS"
    llm = FakeLLM(GOOD)
    d = extract_resume(text, llm, max_chars=1000)
    sent = llm.calls[0]
    assert "HEAD-JOBS" in sent and "TAIL-SKILLS" in sent and "omitted" in sent
    assert len(sent) < 1200
    assert any("middle was omitted" in w for w in d.warnings)


def test_default_cap_is_generous():
    assert MAX_CHARS >= 60_000
    llm = FakeLLM(GOOD)
    extract_resume("a" * 40_000, llm)  # well under the cap: untouched, no warning
    assert "omitted" not in llm.calls[0]
