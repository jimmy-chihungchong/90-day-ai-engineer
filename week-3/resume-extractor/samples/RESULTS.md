# Test run: 10 synthetic resumes

Run with `python scripts/run_samples.py [--backend nim] [--model ...]`
(ground truth in `samples/expected.json`; regenerate the files with `python samples/make_samples.py`).

| # | File | What it tests | v1 (before fixes) | v2 (after fixes) |
|---|------|---------------|-------------------|------------------|
| 01 | 01_simple.docx | Baseline Word, month-name dates, a gap | PASS | PASS |
| 02 | 02_simple.pdf | Baseline PDF, `MM/YYYY` dates | PASS | PASS |
| 03 | 03_two_column.pdf | Sidebar + main column; extracted text interleaves | PASS | PASS |
| 04 | 04_tables_header.docx | Name only in header; jobs/skills in tables | PASS | PASS |
| 05 | 05_scanned.pdf | Image-only PDF (no text layer) | PASS (clear "scanned" error) | PASS |
| 06 | 06_spanish.docx | Spanish months, "actualidad" | PASS | PASS |
| 07 | 07_long.pdf | ~35k chars, 12 jobs, skills at the end | **FAIL** | PASS |
| 08 | 08_messy_dates.docx | Overlap, gap, "Summer 2017", year-only | **FAIL** | PASS |
| 09 | 09_creative.pdf | `'21 - now` dates, skills written as a sentence | PASS | PASS |
| 10 | 10_student.docx | No jobs; projects must not count | PASS | PASS |

| Backend / model | v1 | v2 |
|---|---|---|
| LM Studio `openai/gpt-oss-20b` | 8 / 10 | **10 / 10** (4-17s per resume) |
| NIM `openai/gpt-oss-20b` | 8 / 10 | **10 / 10** (8-26s) |
| NIM `nvidia/nemotron-3.5-lightning-30b-a3b` | 8 / 10 | **10 / 10**, but slow: 07 took 129s and 08 took 108s (reasoning), 07 needed the JSON retry |

## What was fixed (v1 -> v2)

1. **Truncation (07).** Cap raised from 24,000 to 60,000 characters (`MAX_RESUME_CHARS`). When a resume is still
   too long, the first 70% and last 30% are kept (jobs at the top, skills at the end) instead of cutting the tail,
   and a warning says the middle was omitted.
2. **Vague dates (08).** The prompt now tells the model to copy season dates verbatim, and the date parser
   understands spring/summer/fall/autumn/winter ("Summer 2017" = Jun-Aug, about 3 months).
3. **Token budget.** `max_tokens` raised from 4096 to 8192 (`LLM_MAX_TOKENS`); if a reasoning model still returns an
   empty answer because it used the whole budget, the request is retried once with double. This fixed the Nemotron
   failures on 03 and 07 (v1: empty reply, `finish_reason=length`).

## Still breaks / not covered

- **LM Studio context window.** The local gpt-oss-20b is loaded with an 8192-token context (model max is 131072).
  A resume that does not fit fails with a clear "does not fit in the context window" message. Stress test with
  random-word text (about 2 characters per token, a worst case): 20k characters (10,285 tokens) already failed. Real English is closer to 4
  chars/token, so roughly 20-25k characters of normal text is the practical local limit until the context length
  is raised in LM Studio (or set `MAX_RESUME_CHARS=20000`).
- **Year-only ranges** (`2014 - 2016`) are still counted as whole calendar years and can overstate experience.
- **Scanned PDFs**: rejected, no OCR. **`.doc`** and password-protected files: rejected.
- **Non-English dates** only worked because the model normalized them to `YYYY-MM`; the Python parser knows English
  months and seasons only.
- One run per model at temperature 0 on synthetic resumes (cleaner than real ones). Not covered: real Canva/LinkedIn
  exports, other models, the Streamlit Cloud deployment.

## Notes on the NIM catalog (2026-10-23, free-tier key)

Of 14 catalog models probed with a chat call, 5 worked (`openai/gpt-oss-20b`, `minimaxai/minimax-m3.1-flash`,
`nvidia/nemotron-3.5-lightning-30b-a3b`, `nvidia/nemotron-3-ultra-550b-a55b`, `z-ai/glm-5.3`), 8 returned
"not found" and 3 timed out. Catalog presence does not imply availability; `scripts/check_nim.py` tests it.
