# Token & Cost Visualizer (prototype)

A Streamlit app for learning how LLMs work without the maths. Paste text and see how it splits into
tokens and what it would cost across models, then explore context windows, temperature and sampling,
why models hallucinate, and what embeddings are. It runs against a local LM Studio server or NVIDIA NIM.

## Run locally (Windows)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # defaults work for LM Studio
streamlit run app.py
```
Or double-click `RUN.BAT` once the venv exists.

Requirements: Python 3.11+ (developed on 3.14). For the Temperature, Hallucination and Embeddings tabs,
LM Studio's server must be running on `localhost:1234` with `llama-3.2-3b-instruct` (the default chat model)
and `text-embedding-nomic-embed-text-v1.5` loaded. The Tokens & cost and Context window tabs need no server.
The first run downloads tokenizer files, so it needs internet once.

## The tabs
| Tab | What it teaches | Needs a model? |
|---|---|---|
| Tokens & cost | Colored token chips, token IDs, and estimated cost per model. A slider sets the expected reply length, because output tokens cost more than input. | No |
| Context window | A window is shared by your prompt and the reply. Shows what fits and what the model never sees. | No |
| Temperature | The same prompt at three temperatures plus top_p, with a chart of how the odds for the first word get reshaped. | Chat |
| Hallucination | Each word shaded by the model's confidence. Made-up details show up as low-confidence words. | Chat |
| Embeddings | Sentences become 768 numbers. A 2D map shows similar meanings clustering, plus search by meaning. | Embedding |

## Structure
| File | Role |
|---|---|
| `app.py` | Streamlit entry point: tabs and the Tokens & cost tab |
| `tokviz/config.py` | `.env` config and the local / NIM backend switch |
| `tokviz/tokenizer.py` | tiktoken wrapper: text to tokens |
| `tokviz/pricing.py` + `data/pricing.json` | Model prices, context windows and cost maths |
| `tokviz/render.py` | Token "chips" and markdown escaping |
| `tokviz/llm.py` | OpenAI-compatible client: sampling params, logprobs, timeout, suggests another model |
| `tokviz/sampling.py` | Pure maths for temperature and top_p |
| `tokviz/confidence.py` | Confidence bands and summary from logprobs |
| `tokviz/context.py` | What fits in a context window |
| `tokviz/embeddings.py` | Embedding calls, cosine similarity, 2D projection (numpy only) |
| `tokviz/ui_common.py` | Shared model picker and the slow-model warning |
| `tokviz/tab_*.py` | One file per LLM-backed or interactive tab |

## Tests
```powershell
pytest
```
63 tests. The ones that call LM Studio or NIM skip themselves automatically if that server isn't available
(no LM Studio running, or no `NIM_API_KEY`), so `pytest` works anywhere. `tests/test_app.py` drives the whole UI headlessly with Streamlit's `AppTest`.

## Prices
`data/pricing.json` holds USD per 1M tokens, fetched from the vendors' pricing pages on **2026-10-13**
(the date is stored in the file). Prices change, so check before relying on them and edit the file freely.
Local models are $0. Paid models' context windows are `null` (not verified), so the Context window tab
offers the local models plus a custom demo window.

## Configuration (`.env`)
See `.env.example`. Key settings: `BACKEND` (`local` or `nim`), `LMSTUDIO_BASE_URL`, `LMSTUDIO_CHAT_MODEL`,
`LMSTUDIO_EMBED_MODEL`, `NIM_API_KEY`, `NIM_CHAT_MODEL`, `NIM_EMBED_MODEL`, `LLM_TIMEOUT_SECONDS`.
Never commit `.env`.

## Online models (NVIDIA NIM)
Pick **Online (NVIDIA NIM)** in the sidebar (needs `NIM_API_KEY`, free at build.nvidia.com). NIM lists ~80
models but most return 404, so the app offers a short list verified with a real key on **2026-10-13**:

| Model | Use | Notes |
|---|---|---|
| `nvidia/ising-calibration-1.5-31b` (default) | Chat | About 1s, no hidden reasoning, returns logprobs. It is a qubit-calibration specialist, so it sometimes answers off-topic questions about itself. |
| `nvidia/nemotron-3-super-120b-a12b` | Chat | Fast, but a reasoning model |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | Chat | Fast, reasoning, occasional 503 (capacity) |
| `openai/gpt-oss-20b` | Chat | Slow here (10-18s), reasoning |
| `nvidia/nemotron-3-embed-1b` (default) | Embeddings | 2048 numbers per text. Needs `input_type` (the app sends it). |

Availability changes: `nvidia/nv-embedqa-e5-v5` (the old default) was retired on 2026-08-25 and now returns
410. Override the list with `NIM_CHAT_MODELS` and the embedder with `NIM_EMBED_MODEL`. If your `.env` was
copied from an older `.env.example`, update `NIM_CHAT_MODEL` and `NIM_EMBED_MODEL` to the values above.

**Reasoning models on NIM** (gpt-oss, Nemotron) spend the tabs' small token budgets on hidden thinking, so
they often return an empty answer, and their logprobs include the thinking tokens. The Hallucination tab
detects an empty answer and says so. Use the first model in the list for the demos.

## Deploy (Streamlit Community Cloud) - not done yet
LM Studio isn't reachable from the cloud, so deploy with NIM: push the repo and set `BACKEND = "nim"`,
`NIM_API_KEY` and (optionally) the model names in the app's Secrets. With `BACKEND=nim` the LM Studio option is
hidden. NIM itself has been tested locally with a real key; **the Streamlit Cloud deployment itself has not**.
Secrets should reach the app as environment variables, but that is unverified here.

## Notes
- Token counts are exact only for OpenAI-family tokenizers (gpt-oss). Other models are approximated with
  `o200k_base` and marked with "≈".
- If a model gives no answer within `LLM_TIMEOUT_SECONDS` (default 45), the UI warns and offers a one-click
  switch to another model.
- The Temperature chart uses the model's real probabilities for the first word. LM Studio reports them before
  temperature is applied, so the app reshapes them with the standard formulas. Only the top 8 candidates are
  shown, renormalised among themselves.
- Well-trained models refuse to answer unknown questions. The Hallucination tab's "pressure the model" box adds a
  system prompt telling it never to say "I don't know", which is what makes it invent details.
- Local embeddings (nomic) use no task prefix. Nomic's docs recommend prefixes, but without them similarity
  scores spread out more and are easier to read. The rankings are the same. NIM's embedder does need a hint.
- Reasoning models (such as gpt-oss) spend tokens thinking, so very small `max_tokens` can give an empty answer.

## Manual test cases
Run these on the app. Cases marked **(L)** need the local LM Studio setup.

| # | Area | Steps | Expected |
|---|---|---|---|
| 1 | Tokenize | Type "Hello world, hello tokens!" in the Tokens & cost text box and click **Tokenize**. | Nothing changes until you click. After clicking, the count is 6 and one chip appears per token. |
| 2 | Whitespace | Paste text with two lines and several spaces. | `·` marks spaces and `↵` marks newlines on the chips. |
| 3 | Cost slider | Move "Expected output length" from 500 to 4000. | The blue banner and the table's Output $ and Total $ columns rise. Input $ does not change. |
| 4 | Odd text | Paste emoji, Chinese text, and `<\|endoftext\|>`. | No error. Emoji and CJK use more tokens, and the special token is treated as plain text. |
| 5 | Context window | Open Context window with the demo text and the default 120-token window. | A warning says tokens don't fit. The secret code sentence appears in the red "Cut off" box and the question stays visible. |
| 6 | Keep strategy | Switch to "Keep the beginning". | The question at the end is now cut, and the secret code stays. |
| 7 | Temperature (L) | Open Temperature and click **Generate**. | Three outputs appear with timings, and a bar chart where low temperature piles probability on one word. |
| 8 | Hallucination (L) | Choose "Made-up person" with the pressure box ticked and click **Ask**. | An invented name appears with red or orange shading on the name, and "Shaky words" is above 0. |
| 9 | Refusal (L) | Repeat case 8 with the pressure box unticked. | The model says it can't verify the answer, instead of inventing one. |
| 10 | Embeddings (L) | Open Embeddings, click **Embed and map**, then search "a pet resting at home". | Four topic clusters appear, and the top matches are the cat and dog sentences. |
| 11 | Timeout (L) | Set `LLM_TIMEOUT_SECONDS=1` in `.env`, restart, and click **Generate** with `openai/gpt-oss-20b` selected. | A warning appears with a "Switch to ..." button, and clicking it changes the model. |
| 12 | Server down (L) | Stop the LM Studio server, wait 30 seconds (the model list is cached), and reload the Temperature tab. | A red "Cannot reach ... Is the server running?" message replaces the model picker, with no stack trace. |
| 13 | NIM switch | Choose **Online (NVIDIA NIM)** in the sidebar, then run Generate, Ask and Embed and map. | The model list shows the verified NIM models, and all three tabs work (answers take a few seconds). |
