# Chat with your PDF

A local, self-correcting RAG chatbot. Upload PDFs and/or add web page URLs, then ask questions about them. Everything runs on your machine: the chat and embedding models are served by [LM Studio](https://lmstudio.ai/), and the UI is a [Streamlit](https://streamlit.io/) app.

This is a home practice project from week 5 of my 90-day AI engineer plan (RAG, chunking, vector stores, retrieval grading).

## How it works

The pipeline is a self-RAG graph built with [LangGraph](https://langchain-ai.github.io/langgraph/):

```
START -> retrieve -> grade_documents -+-> generate -+-> END   (grounded and useful)
             ^                        |             |
             |                        |             +-> generate         (not grounded: retry)
             +------ transform_query <+             +-> transform_query  (doesn't answer the question)
                     (no relevant docs)             +-> END              (retry limit reached)
```

1. **Retrieve:** fetch the top chunks from an in-memory Chroma vector store.
2. **Grade documents:** an LLM grader drops chunks that aren't relevant to the question.
3. **Generate:** answer from the remaining chunks, in three sentences or fewer.
4. **Check the answer:** a hallucination grader checks the answer is grounded in the chunks, and an answer grader checks it addresses the question.
5. **Retry or rewrite:** an ungrounded answer is regenerated. An answer that misses the question, or a retrieval with no relevant chunks, triggers a query rewrite and a new retrieval.
6. **Give up:** after `MAX_RETRIES` (2) generation attempts the last answer is returned with a warning.

Sources are split into 250-token chunks and indexed together, so one question searches every PDF and URL at once.

## Features

- Multiple PDFs (PyMuPDF) and web pages (`WebBaseLoader`) in one index
- Three Lilian Weng blog posts are pre-loaded as web sources in every new session; edit them in the sidebar
- Question box directly under the title, plus two one-click test questions
- Live progress while the graph runs, and the source chunks shown under each answer (file and page for PDFs, page title for web pages)
- A failed URL is skipped with a warning instead of breaking the index

## Requirements

- Python 3.10+ (developed on 3.14)
- LM Studio running its local server, with a chat model and an embedding model loaded

## Setup

```powershell
# 1. Create a virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Configure LM Studio settings
copy .env.example .env
#    then edit .env (see below)

# 3. Run the app
.\.venv\Scripts\streamlit.exe run app.py
```

Use the virtual environment. Installing into a global Python can clash with the older pinned versions in the notebook's install cell (the newer `langgraph` needs a newer `langchain-core`).

### Configuration (`.env`)

| Variable | Meaning |
|---|---|
| `LMSTUDIO_BASE_URL` | LM Studio's OpenAI-compatible endpoint, e.g. `http://localhost:1234/v1` |
| `LMSTUDIO_CHAT_MODEL` | Chat model identifier, e.g. `openai/gpt-oss-20b` |
| `LMSTUDIO_EMBED_MODEL` | Embedding model identifier, e.g. `text-embedding-nomic-embed-text-v1.5` |

`.env` is gitignored. `.env.example` also lists optional LangSmith and OpenAI-style variables; the app only needs the three above.

## Usage

1. Start LM Studio's server and load both models.
2. Run the app and open the URL Streamlit prints (default http://localhost:8501).
3. In the sidebar, upload PDFs and/or edit the URL list (one URL per line). The index rebuilds whenever this set changes.
4. Type a question, or click one of the test-question buttons.

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: sidebar sources, chat, source display |
| `rag_graph.py` | Loaders, retriever, graders and the LangGraph graph (no Streamlit code) |
| `chat-with-your-pdf.ipynb` | The original notebook the app was built from, with the step-by-step experiments |
| `requirements.txt` | Dependencies for the app |
| `.env.example` | Template for the LM Studio settings |

## Limitations

- **No conversation memory.** Each question is answered independently, so follow-ups like "what about the second one?" don't work.
- **Slow retries.** A rejected answer costs extra LLM calls, which is noticeable with a local model. Lower `MAX_RETRIES` in `rag_graph.py` to speed it up.
- **Index is in memory.** The vector store is rebuilt when the app restarts or the sources change; nothing is persisted.
- **One shared index.** Retrieval pulls from all sources together, so a question about one document can return chunks from another.

## Reference

Based on the LangGraph self-RAG example: https://github.com/langchain-ai/langgraph/tree/main/examples/rag/langgraph_self_rag.ipynb
