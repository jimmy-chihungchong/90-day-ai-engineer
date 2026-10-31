# Semantic Notes Search (prototype)

Search your own notes, bookmarks or journal **by meaning**. Ask "what did I write about burnout?" and get entries about
exhaustion, overwork or running on fumes, even if the word never appears.

Prototype scope: no auth, no database, no PDF support. Built to show the full workflow end to end.

## How it works

```
 .md / .txt / .json ──► loaders.py ──► chunks (heading / record / paragraph, ≤1200 chars)
                                           │
                       embeddings.py ◄─────┘   OpenAI-compatible /v1/embeddings
                          │                    (LM Studio nomic, or NVIDIA NIM)
                          ▼
                numpy cosine index  (+ .index/ cache keyed by chunk-text hash, so re-builds only embed new text)
                          │
 question ──► embed ──► top-k chunks ──► llm.py (gpt-oss-20b) ──► short cited answer
                                          └─ timeout? UI warns you and lists other models to try
```

| Module | Job |
|---|---|
| `semsearch/loaders.py` | Parse MD (split on headings), TXT (paragraphs), JSON (one chunk per record) |
| `semsearch/embeddings.py` | Embedding client, nomic task prefixes, cosine index, disk cache |
| `semsearch/llm.py` | Answer from retrieved excerpts, timeout handling, alternative-model suggestions |
| `app.py` | Streamlit UI |
| `cli.py` | Terminal test harness |

## Run locally (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

In LM Studio, start the local server (port 1234) and load `text-embedding-nomic-embed-text-v1.5` and `openai/gpt-oss-20b`.
Keep the LM Studio context length at 8k or more; the app caps the excerpts it sends to about 9,000 characters.

Terminal test:

```bash
python cli.py ../log "which jobs were a good fit for my RAG experience?" --answer
```

## Deploy on Streamlit Community Cloud

1. Push this folder to GitHub and create the app from `app.py`.
2. Add secrets (App settings > Secrets):
   ```toml
   PROVIDER = "nvidia"
   NVIDIA_API_KEY = "nvapi-..."
   ```
3. A hosted app cannot reach your local files. Use the **upload** box in the sidebar, or commit notes into `sample_notes/`.

## Assumptions

- Notes are UTF-8 text. JSON is a list of records, or an object containing lists of records.
- NVIDIA chat defaults to `nvidia/nemotron-3.5-lightning-30b-a3b` because `openai/gpt-oss-20b` timed out on NIM in testing (set `NVIDIA_CHAT_MODEL` to switch back). For embeddings it uses `nvidia/nemotron-3.5-embed-1b` (change via `NVIDIA_EMBED_MODEL`).
- Vectors from different embedding models are not comparable; the cache is per embedding model.
- The LLM can embellish. Always check the cited excerpts under the answer.

## If the LLM is slow

After `LLM_TIMEOUT_SECONDS` (default 60) the UI shows a warning with a toast, prints `[ALERT]` in the console, and
offers other chat models available on the same endpoint to retry with.
