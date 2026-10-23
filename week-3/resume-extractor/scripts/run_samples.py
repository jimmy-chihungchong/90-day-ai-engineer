"""Run every sample resume through the pipeline and score it against samples/expected.json.

Usage:  python scripts/run_samples.py [--model MODEL] [--backend lmstudio|nim] [--out outputs/run.json]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from resume_extractor.config import get_settings  # noqa: E402
from resume_extractor.pipeline import run_pipeline  # noqa: E402


def score(res, exp) -> list[str]:
    """Return a list of problems (empty = pass)."""
    if exp.get("expect_error"):
        return [] if res.error and "scanned" in res.error.lower() else \
            [f"expected a 'scanned file' error, got: {res.error or 'success'}"]
    if res.error:
        return [f"pipeline error: {res.error}"]
    d, issues = res.data, []
    if exp["name"].lower() not in (d.name or "").lower():
        issues.append(f"name: got {d.name!r}, want {exp['name']!r}")
    if exp["email"].lower() != (d.email or "").lower():
        issues.append(f"email: got {d.email!r}")
    got = d.total_years_experience
    if exp["years"] is None:
        if got not in (None, 0):
            issues.append(f"years: got {got}, want none (no jobs)")
    elif got is None or abs(got - exp["years"]) > exp["tol"]:
        issues.append(f"years: got {got}, want {exp['years']} (±{exp['tol']})")
    have = " | ".join(s.lower() for s in d.skills)
    missing = [s for s in exp["skills"] if s.lower() not in have]
    if missing:
        issues.append(f"skills missing: {missing}")
    if len(d.experience) != exp["jobs"]:
        issues.append(f"jobs: got {len(d.experience)}, want {exp['jobs']}")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="lmstudio")
    ap.add_argument("--model", default="")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "run.json"))
    a = ap.parse_args()

    ov = {"LLM_BACKEND": a.backend, "LLM_TIMEOUT_SECONDS": "180"}
    if a.model:
        ov["LMSTUDIO_MODEL" if a.backend == "lmstudio" else "NIM_MODEL"] = a.model
    settings = get_settings(ov)
    expected = json.loads((ROOT / "samples" / "expected.json").read_text(encoding="utf-8"))

    report, passed = [], 0
    print(f"model={settings.model} backend={settings.backend}\n")
    for name, exp in expected.items():
        res = run_pipeline(name, (ROOT / "samples" / "resumes" / name).read_bytes(), settings)
        issues = score(res, exp)
        passed += not issues
        print(f"{'PASS' if not issues else 'FAIL'}  {name:26s} {res.seconds:5.1f}s")
        for i in issues:
            print(f"        - {i}")
        for w in (res.data.warnings if res.data else []):
            print(f"        ! warning: {w}")
        report.append(dict(file=name, passed=not issues, issues=issues, seconds=round(res.seconds, 1),
                           note=exp["note"], error=res.error, timed_out=res.timed_out,
                           data=res.data.model_dump() if res.data else None))
    print(f"\n{passed}/{len(expected)} passed")
    out = Path(a.out); out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": settings.model, "results": report}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
