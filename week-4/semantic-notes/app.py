import os
from pathlib import Path

import streamlit as st

# Streamlit Cloud: copy secrets into env so config.py sees them
try:
    for k, v in st.secrets.items():
        if isinstance(v, str):
            os.environ.setdefault(k, v)
except Exception:
    pass

from semsearch.config import get_settings
from semsearch.embeddings import Embedder, Index
from semsearch.llm import LLMTimeout, answer, suggest_alternatives
from semsearch.loaders import SUPPORTED, load_folder, load_text

st.set_page_config(page_title="Semantic Notes Search", page_icon="🔎", layout="wide")
st.title("🔎 Semantic Notes Search")
st.caption("Ask in your own words — matches by meaning, not keywords.")

with st.sidebar:
    default_provider = os.getenv("PROVIDER", "lmstudio").lower()
    provider = st.radio("Backend", ["lmstudio", "nvidia"], index=["lmstudio", "nvidia"].index(default_provider))
    s = get_settings(provider)
    st.text(f"chat:  {s.chat_model}\nembed: {s.embed_model}")
    if provider == "nvidia" and not s.api_key:
        st.error("Set NVIDIA_API_KEY in .env or Streamlit secrets.")
    folder = st.text_input("Notes folder", os.getenv("NOTES_DIR", "./sample_notes"))
    uploads = st.file_uploader("...or upload files", type=[e.strip(".") for e in SUPPORTED], accept_multiple_files=True)
    top_k = st.slider("Results", 3, 15, 6)
    use_llm = st.checkbox("Summarise answer with LLM", value=True)
    build = st.button("Build / refresh index", type="primary")

if build or "index" not in st.session_state or st.session_state.get("provider") != provider:
    chunks = []
    if folder and Path(folder).is_dir():
        chunks += load_folder(folder)
    for f in uploads or []:
        chunks += load_text(f.name, f.getvalue().decode("utf-8", errors="replace"))
    if not chunks:
        st.info("No notes found. Point to a folder or upload .md / .txt / .json files, then click Build.")
        st.stop()
    bar = st.progress(0.0, text=f"Embedding {len(chunks)} chunks (cached chunks are skipped)...")
    try:
        embedder = Embedder(s)
        st.session_state.index = Index.build(chunks, embedder, progress=lambda f: bar.progress(f))
        st.session_state.provider = provider
    except Exception as e:
        bar.empty()
        st.error(f"Embedding failed: {e}\n\nIs LM Studio running with the embedding model loaded (or is the NVIDIA key valid)?")
        st.stop()
    bar.empty()

index: Index = st.session_state.index
st.sidebar.success(f"{len(index.chunks)} chunks indexed")

query = st.text_input("Ask your notes", placeholder="e.g. what did I write about burnout?")
if query:
    hits = index.search(Embedder(s).embed([query], "query")[0], top_k)

    if use_llm:
        model = st.session_state.get("model_override") or s.chat_model
        with st.spinner(f"Asking {model}..."):
            try:
                st.markdown("### Answer\n" + answer(s, query, hits, model))
            except LLMTimeout:
                alts = suggest_alternatives(s)
                st.toast(f"{model} is not responding", icon="⏰")
                print(f"[ALERT] {model} timed out after {s.timeout}s")
                st.warning(f"⏰ **{model}** did not answer within {s.timeout:.0f}s. Suggested alternatives: "
                           + (", ".join(f"`{a}`" for a in alts) or "none found") + ". Pick one below and re-run.")
                if alts:
                    st.session_state.model_override = st.selectbox("Try another model", alts)
            except Exception as e:
                st.error(f"LLM error: {e}")

    st.markdown("### Matching entries")
    for score, c in hits:
        with st.expander(f"{score:.2f} · {c.source} · {c.title}"):
            st.text(c.text)
