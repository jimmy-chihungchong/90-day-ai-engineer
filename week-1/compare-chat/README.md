# Compare Chat (prototype)

A duck.ai-style chatbot with conversation memory, a two-model switch, and a side-by-side
compare mode. Runs against local LM Studio or NVIDIA NIM. Memory can be saved to / loaded from JSON.

## Run locally (Windows)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # defaults work for LM Studio
streamlit run app.py
```
Requirements: LM Studio server on `localhost:1234` with `microsoft/phi-4-mini-reasoning`
and `llama-3.2-3b-instruct` loaded.

## Structure
| File | Role |
|---|---|
| `app.py` | Streamlit UI |
| `chatbot/config.py` | .env / Streamlit-secrets config, backend + model registry |
| `chatbot/llm_client.py` | OpenAI-compatible streaming client, timeout -> suggests the other model |
| `chatbot/memory.py` | Per-model history, context trimming, JSON export/import |
| `chatbot/compare.py` | Ask both models (sequential on LM Studio, parallel on NIM) |

## Tests
```powershell
pytest -m "not live"   # no server needed
pytest                 # includes live LM Studio tests
```

## Deploy (Streamlit Community Cloud)
Push the repo, set `BACKEND = "nim"`, `NIM_API_KEY`, and (if needed) `NIM_MODEL_A/B`
in the app's Secrets. LM Studio is not reachable from the cloud, so use NIM there.

## Notes
- Each model keeps its own history, so compare stays fair.
- If no token arrives within `LLM_TIMEOUT_SECONDS` (default 45), the UI warns and offers a one-click switch to the other model.
- NIM model IDs are best guesses; confirm them in the NIM catalog.
- Memory JSON contains message text, timestamps and latency. Treat it as private.
