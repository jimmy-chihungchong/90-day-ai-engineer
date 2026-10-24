# Resume to JSON (prototype)

Upload a messy **PDF** or **Word (.docx)** resume and get clean, structured JSON: name, contact details,
skills, work history, education and total years of experience. The app shows the raw extracted text next to
the JSON so you can see how scattered data becomes organized data.

This is a **prototype**: a quick, modular, end-to-end demo, not production software.

```
Upload ─▶ parsers.py ─▶ raw text ─▶ extractor.py ─▶ llm.py ─▶ JSON ─▶ schema.py ─▶ experience.py ─▶ UI
 PDF/DOCX  pdfplumber                 prompt +        LM Studio        validate     years computed
           python-docx                1 retry         or NVIDIA NIM                 in Python
```

| Module | File | Job |
|---|---|---|
| 1 | `src/resume_extractor/parsers.py` | PDF/DOCX bytes → text (tables and Word headers included) |
| 2 | `src/resume_extractor/schema.py`, `experience.py` | Output model; years of experience from date ranges (overlaps merged) |
| 3 | `src/resume_extractor/llm.py`, `config.py` | OpenAI-compatible client, backend switch, timeout alert + model suggestions |
| 4 | `src/resume_extractor/extractor.py` | Prompt, JSON validation, one repair retry |
| 5 | `app.py`, `pipeline.py` | Streamlit UI |

**Design choice:** the LLM only *extracts* (names, skills, dated jobs). Total years of experience is
**computed in Python** from the job dates, because LLMs are unreliable at date arithmetic.

## Quick start (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt      # requirements.txt is enough to just run the app
copy .env.example .env
streamlit run app.py
```

Then open http://localhost:8501 and upload a file (try `samples/resumes/`).

### Local backend: LM Studio

1. Start the LM Studio server (default `http://localhost:1234`) and load `openai/gpt-oss-20b`.
2. Keep `LLM_BACKEND=lmstudio` in `.env`. The sidebar lists whatever chat models LM Studio has loaded.

### Hosted backend: NVIDIA NIM

1. Get a key at <https://build.nvidia.com> (free trial credits, rate limited).
2. Put it in `.env`: `LLM_BACKEND=nim` and `NIM_API_KEY=nvapi-...` (or paste it in the sidebar; it stays in the session).
3. See which models your key can actually call:

   ```powershell
   python scripts/check_nim.py
   ```

   `openai/gpt-oss-20b` is the default. **Being listed in the NIM catalog does not mean a model is callable**:
   on 2026-10-23, 8 of 14 catalog models I tried returned "not found" and 3 timed out. These answered a test
   request on a free key and are the UI's fallback suggestions (`config.py`):
   `minimaxai/minimax-m3.1-flash`, `nvidia/nemotron-3.5-lightning-30b-a3b`,
   `nvidia/nemotron-3-ultra-550b-a55b`, `z-ai/glm-5.3`. Availability changes, so re-run `check_nim.py`.

   Full-suite results on NIM are in [`samples/RESULTS.md`](samples/RESULTS.md).

### Timeouts

If the model has not answered within `LLM_TIMEOUT_SECONDS` (default 90) the UI shows a warning with
one-click buttons to retry on other models (loaded LM Studio models, or the NIM fallbacks).

## Deploy on Streamlit Community Cloud

1. Push the repo to GitHub. Main file path: `week-3/resume-extractor/app.py` (adjust to your layout).
2. In **Advanced settings → Secrets**, add:

   ```toml
   LLM_BACKEND = "nim"
   NIM_API_KEY = "nvapi-..."
   NIM_MODEL = "openai/gpt-oss-20b"
   ```

3. Deploy. `requirements.txt` holds only runtime dependencies (no pytest/reportlab).

Never commit `.env` or `.streamlit/secrets.toml` (both are git-ignored). Uploaded resumes are sent to the
selected LLM provider, so do not use real candidate data with a hosted backend unless that is acceptable.

## Output

```json
{
  "name": "Maria Gonzalez-Lee",
  "email": "maria.lee@example.com",
  "phone": "(415) 555-0142",
  "location": "San Francisco, CA",
  "links": ["linkedin.com/in/marialee"],
  "summary": "Backend engineer who likes data pipelines.",
  "skills": ["Python", "Go", "SQL", "Airflow"],
  "experience": [
    {"company": "Northwind Analytics", "title": "Staff Engineer", "start": "2021-03", "end": "Present",
     "description": "Led migration of ETL jobs to Airflow."}
  ],
  "education": [{"institution": "UC Davis", "degree": "B.S. Computer Science", "start": "2011", "end": "2015"}],
  "total_years_experience": 11.8,
  "warnings": []
}
```

`total_years_experience` is the union of all job date ranges (overlapping jobs count once, gaps are not
counted). Jobs with unusable dates are skipped and listed in `warnings`.

## Tests

```powershell
pytest                                   # 52 unit tests, no LLM needed
python samples/make_samples.py           # regenerate the 10 synthetic resumes + answer key
python scripts/run_samples.py            # run them through the real LLM and score the output
```

Last results (details in [`samples/RESULTS.md`](samples/RESULTS.md)): **10 / 10** on `openai/gpt-oss-20b` via both
LM Studio and NIM, and 10 / 10 on NIM `nvidia/nemotron-3.5-lightning-30b-a3b` (slow, see the file). The first run scored 8 / 10;
the two failures (long resume truncation, "Summer 2017" dates) and a too-small token budget were then fixed.

### Where it still breaks (known limitations)

| Problem | Effect |
|---|---|
| LM Studio's context window (8192 tokens for gpt-oss-20b by default) | Resumes over roughly 20-25k characters fail with a clear "does not fit" error. `.env.example` sets `MAX_RESUME_CHARS=20000` so the head and tail are kept instead; raise it for NIM, or raise Context Length in LM Studio |
| Resume text > `MAX_RESUME_CHARS` (default 60,000) | The middle is omitted (head 70% / tail 30% kept) and a warning is shown |
| Year-only ranges ("2014 - 2016") | Counted as whole calendar years, so experience can be overstated |
| Scanned / image-only PDFs | Rejected with a clear message; no OCR |
| Legacy `.doc`, password-protected files | Rejected |
| Two-column PDFs | Extracted text is interleaved; worked on gpt-oss-20b, likely worse on small models |
| Non-English dates | Worked only because the LLM normalized them to `YYYY-MM`; the Python parser knows English months and seasons only |
| Reasoning models on NIM (e.g. Nemotron) | Correct but slow (up to ~2 minutes) because the token budget is doubled when they run out |

### Manual checks worth doing in the UI

- Upload each file in `samples/resumes/` and compare the raw text with the JSON.
- Set the timeout to 10s and use a large model to see the timeout alert and the retry buttons.
- Switch to NIM with no key to see the missing-key behaviour.

## Gotchas found while building

- **LM Studio + gpt-oss-20b rejects `system` messages** with a server-side "peg-native format" error, so for the
  LM Studio backend the system prompt is folded into the user message (`llm.py`).
- **LM Studio rejects `response_format: {"type": "json_object"}`** (only `json_schema` or `text`). The app uses
  plain prompting, tolerant JSON extraction and one validation retry, which works on both backends.
- The `openai` 3.x package depends on `httpx2`, not `httpx`; use `openai.Timeout` rather than importing httpx.
- Reasoning models spend tokens thinking: `LLM_MAX_TOKENS` defaults to 8192, an empty reply is retried once with double, and a second empty reply raises a clear error.
- LM Studio loads models with a small context window by default; see `MAX_RESUME_CHARS` in `.env.example`.

## Project layout

```
app.py                    Streamlit UI
src/resume_extractor/     parsers, schema, experience, llm, config, extractor, pipeline
scripts/                  run_samples.py (scorer), check_nim.py (NIM model probe)
samples/                  make_samples.py, expected.json, resumes/, RESULTS.md
tests/                    pytest unit tests
.env.example              settings template
```

## Ideas for next steps

Chunk very long resumes instead of dropping the middle, midpoint convention for year-only dates, OCR fallback
for scanned PDFs, skill normalization with the local `text-embedding-nomic-embed-text-v1.5` model
("JS" → "JavaScript"), constrained decoding via `json_schema` on LM Studio.
