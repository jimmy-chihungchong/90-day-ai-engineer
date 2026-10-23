from resume_extractor.config import Settings
from resume_extractor.llm import LLMTimeout
from resume_extractor.pipeline import run_pipeline
from tests.test_parsers import BODY, make_docx

S = Settings("lmstudio", "http://x/v1", "m1", "k", 5)
GOOD = '{"name": "Jane Doe", "skills": ["Python"], "experience": []}'


class Fake:
    def __init__(self, reply=None, exc=None): self.reply, self.exc = reply, exc
    def complete(self, system, user):
        if self.exc: raise self.exc
        return self.reply


def test_success():
    r = run_pipeline("a.docx", make_docx(BODY), S, Fake(GOOD))
    assert r.error is None and r.data.name == "Jane Doe" and "Jane Doe" in r.text and r.seconds >= 0


def test_timeout_carries_suggestions():
    r = run_pipeline("a.docx", make_docx(BODY), S, Fake(exc=LLMTimeout("slow", ["m2"])))
    assert r.timed_out and r.suggestions == ["m2"] and r.text  # raw text still shown


def test_parse_error_and_bad_json():
    assert "Unsupported" in run_pipeline("a.txt", b"x", S, Fake(GOOD)).error
    r = run_pipeline("a.docx", make_docx(BODY), S, Fake("nope"))
    assert "valid JSON" in r.error and r.raw_reply == "nope"
