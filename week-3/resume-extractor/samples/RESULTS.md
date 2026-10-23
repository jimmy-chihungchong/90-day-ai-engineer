# Test run: 10 synthetic resumes

Model: `openai/gpt-oss-20b` via LM Studio, run with `python scripts/run_samples.py`
(ground truth in `samples/expected.json`, regenerate files with `python samples/make_samples.py`).

**8 / 10 pass.**

| # | File | What it tests | Result | Time |
|---|------|---------------|--------|------|
| 01 | 01_simple.docx | Baseline Word, month-name dates, a gap | PASS | 4.5s |
| 02 | 02_simple.pdf | Baseline PDF, `MM/YYYY` dates | PASS | 4.3s |
| 03 | 03_two_column.pdf | Sidebar + main column; text extraction interleaves | PASS | 6.4s |
| 04 | 04_tables_header.docx | Name only in header; jobs/skills in tables | PASS | 5.8s |
| 05 | 05_scanned.pdf | Image-only PDF (no text layer) | PASS (clear "scanned" error, no OCR) | 0.0s |
| 06 | 06_spanish.docx | Spanish months, "actualidad" | PASS | 4.4s |
| 07 | 07_long.pdf | ~35k chars, 12 jobs, skills at the end | **FAIL** | 13.2s |
| 08 | 08_messy_dates.docx | Overlap, gap, "Summer 2017", year-only | **FAIL** | 7.2s |
| 09 | 09_creative.pdf | `'21 - now` dates, skills written as a sentence | PASS | 5.7s |
| 10 | 10_student.docx | No jobs; projects must not count | PASS | 4.5s |

## Where it breaks

1. **Long resumes are truncated (07).** Text over 24,000 chars is cut off. Only 9 of 12 jobs were read, so
   experience came out at 18.0 years instead of 24.0, and the Skills section (at the end) was lost entirely.
   The UI shows a truncation warning, but the data is silently incomplete. Fix options: raise the cap
   (gpt-oss-20b has a large context), or chunk the resume and merge.
2. **Season / vague dates inflate experience (08).** "Summer 2017" became `2017`-`2017`, which the calculator
   counts as 12 months instead of ~3, giving 7.5 vs 6.8 years. Year-only ranges (`2014 - 2016`) are
   counted as full calendar years, which overstates by up to a year per role.
3. **Scanned PDFs are rejected (05).** Intended: there is no OCR. It fails with a clear message.

## Things that worked better than expected

- **Two-column PDF (03):** the extracted text is genuinely scrambled (Skills lines mixed into Experience
  lines), yet the model reconstructed everything correctly. This is likely less reliable on smaller models.
- **Spanish (06) and `'21 - now` (09):** the LLM normalized dates to `YYYY-MM` / `Present` itself, so the
  Python date parser never had to understand them. Had the model returned raw text, those jobs would have been
  skipped with a warning.

## Caveats

- One run, one model, temperature 0. Not statistically meaningful; synthetic resumes are cleaner than real ones.
- Not covered yet: real-world files from Canva/LinkedIn exports, password-protected PDFs, `.doc`, other models,
  NVIDIA NIM.

## NVIDIA NIM runs (2026-10-23, free-tier key)

| Model | Pass | Notes |
|---|---|---|
| `openai/gpt-oss-20b` | 8 / 10 | Same two failures (07, 08) as local LM Studio; 10-27s per resume (two suites were running in parallel) |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | 8 / 10 | Different failures: 03 and 07 returned an empty reply with `finish_reason=length`, i.e. the reasoning model spent the whole 4096-token budget thinking. 08 "passed" only because the model left the vague end date null, so the intern job was skipped with a warning. |

Findings:
- **`max_tokens=4096` is too small for some reasoning models** on harder or longer resumes. Not fixed yet.
- **NIM catalog vs reality:** of 14 catalog models probed with a chat call, 5 worked (`gpt-oss-20b` plus the four fallbacks in `config.py`), 8 returned "not found", and 3 timed out (`google/gemma-4-31b-it`, `deepseek-ai/deepseek-v4.1-flash`, `z-ai/glm-5.3-flash`).
