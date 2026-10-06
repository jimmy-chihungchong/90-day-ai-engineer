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
- NIM does not host phi-4-mini-reasoning or llama-3.2-3b-instruct, so NIM uses `openai/gpt-oss-20b` and `meta/muse-glimmer-30b`. Local and online model pairs therefore differ.
- Memory JSON contains message text, timestamps and latency. Treat it as private.

## Manual test cases
Run these on the deployed app. Cases marked **(L)** need the local LM Studio setup.

| # | Area | Steps | Expected |
|---|---|---|---|
| 1 | Basic chat | Send "Say hello in 5 words." | The reply streams in, shows a latency caption, and no error appears. |
| 2 | Memory | Send "My name is Jimmy and I like teal." Then send "What's my name and favourite colour?" | The reply includes both Jimmy and teal. |
| 3 | Model switch | After case 2, switch to the other model and ask "What's my name?" | The new model doesn't know your name, because each model keeps its own history. |
| 4 | Compare | Turn on "Compare both models" and ask "Explain recursion in one sentence." | Two columns appear, each with its own answer and latency. |
| 5 | Compare memory | In compare mode, send "My favourite number is 42." Then send "What is it?" | Both columns recall 42. |
| 6 | Save memory | After cases 2 to 5, click "Save memory (JSON)". | A file downloads. It has a `histories` key, a message per turn, timestamps and latency. |
| 7 | Load memory | Click "New chat", upload the JSON from case 6, then ask "What's my name?" | The old messages reappear and the answer is still correct. |
| 8 | Bad upload | Upload a random JSON, such as `{"foo":1}`, or a non-JSON file renamed to `.json`. | A red "Could not load" message appears and the current chat is untouched. |
| 9 | New chat | Click "New chat" mid-conversation. | The history is cleared, the welcome text returns and the old name is forgotten. |
| 10 | Timeout (L) | Set `LLM_TIMEOUT_SECONDS=1` in `.env` and ask the phi model something long. | A warning appears with a "Switch to llama…" button, and clicking it changes the model. |

Further cases worth trying:
- **Reasoning display (L):** ask phi "What is 17 × 23?" The reasoning should sit in a collapsed "Thinking" block with a clean final answer.
- **Failed backend (L):** stop the LM Studio server and send a message. A red "Cannot reach…" message should appear, with no stack trace.
- **Long conversation:** send around 30 long messages. The app shouldn't crash, and early details may be forgotten because old messages get trimmed.
- **Wrong key:** put a bad `NIM_API_KEY` in the Streamlit secrets. The app should show an error rather than hang.
