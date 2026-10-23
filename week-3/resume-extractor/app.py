"""Module 5: Streamlit UI (upload -> side-by-side raw text / JSON -> download)."""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from resume_extractor.config import NIM_FALLBACKS, get_settings  # noqa: E402
from resume_extractor.llm import LLMClient  # noqa: E402
from resume_extractor.pipeline import run_pipeline  # noqa: E402

st.set_page_config(page_title="Resume to JSON", page_icon=":material/description:", layout="wide")


def secret(key: str) -> str:
    """st.secrets raises if no secrets file exists, so guard it."""
    try:
        return str(st.secrets.get(key, ""))
    except Exception:
        return ""


@st.cache_data(ttl=30, show_spinner=False)
def lmstudio_models(base_url: str) -> list[str]:
    return LLMClient(get_settings({"LLM_BACKEND": "lmstudio", "LMSTUDIO_BASE_URL": base_url})).list_models()


def switch_model(model: str):
    st.session_state.model = model
    st.session_state.run_requested = True


# ---- Sidebar: backend / model --------------------------------------------------------
with st.sidebar:
    st.header("LLM settings")
    default_backend = (secret("LLM_BACKEND") or os.getenv("LLM_BACKEND", "lmstudio")).lower()
    backend = st.segmented_control("Backend", ["lmstudio", "nim"], default=default_backend,
                                   key="backend", format_func=lambda b: {"lmstudio": "LM Studio (local)", "nim": "NVIDIA NIM"}[b])
    backend = backend or default_backend

    overrides = {"LLM_BACKEND": backend, "LLM_TIMEOUT_SECONDS": str(st.session_state.get("timeout", 90))}
    if backend == "lmstudio":
        options = lmstudio_models(get_settings({"LLM_BACKEND": "lmstudio"}).base_url) or ["openai/gpt-oss-20b"]
        model_key = "LMSTUDIO_MODEL"
    else:
        options = ["openai/gpt-oss-20b", *NIM_FALLBACKS]
        model_key = "NIM_MODEL"
        if not (secret("NIM_API_KEY") or os.getenv("NIM_API_KEY")):
            overrides["NIM_API_KEY"] = st.text_input("NVIDIA NIM API key", type="password",
                                                     help="Kept in this session only.")
    current = st.session_state.get("model")
    if current and current not in options:
        options = [current, *options]
    overrides[model_key] = st.selectbox("Model", options, key="model", accept_new_options=True)
    st.number_input("Timeout (seconds)", 10, 600, 90, step=10, key="timeout",
                    help="If the model has not answered by then, you get an alert and a suggested alternative.")

    for k, v in list(overrides.items()):  # secrets win for keys the user did not type
        if not v and secret(k):
            overrides[k] = secret(k)

# ---- Main ----------------------------------------------------------------------------
st.title("Resume to JSON")
st.caption("Upload a messy PDF or Word resume and see it turned into clean, structured data.")

uploaded = st.file_uploader("Resume", type=["pdf", "docx"], label_visibility="collapsed")

if uploaded is None:
    st.info("Upload a PDF or DOCX resume to get started.", icon=":material/upload_file:")
    st.stop()

clicked = st.button("Extract", type="primary", icon=":material/play_arrow:")
if clicked or st.session_state.pop("run_requested", False):
    try:
        settings = get_settings(overrides)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    with st.spinner(f"Asking {settings.model} (up to {settings.timeout:.0f}s)..."):
        st.session_state.result = run_pipeline(uploaded.name, uploaded.getvalue(), settings)

res = st.session_state.get("result")
if res is None or res.filename != uploaded.name:
    st.stop()

if res.error:
    if res.timed_out:
        st.warning(res.error, title="The model is taking too long", icon=":material/hourglass_top:")
    else:
        st.error(res.error, title="Extraction failed")
    if res.suggestions:
        st.write("Try another model:")
        with st.container(horizontal=True):
            for m in res.suggestions:
                st.button(m, key=f"sw_{m}", on_click=switch_model, args=(m,), icon=":material/swap_horiz:")
    if res.raw_reply:
        with st.expander("Raw model reply"):
            st.code(res.raw_reply, language=None)

if res.data:
    d = res.data
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Name", d.name or "—")
    c2.metric("Years of experience", d.total_years_experience if d.total_years_experience is not None else "—")
    c3.metric("Skills found", len(d.skills))
    c4.metric("Time", f"{res.seconds:.1f}s", help=f"Model: {res.model}")
    for w in d.warnings:
        st.warning(w, icon=":material/warning:")

if res.text or res.data:
    left, right = st.columns(2)
    with left:
        st.subheader("Raw extracted text")
        st.text_area("Raw text", res.text, height=520, disabled=True, label_visibility="collapsed")
    with right:
        st.subheader("Structured JSON")
        if res.data:
            payload = res.data.model_dump_json(indent=2)
            st.download_button("Download JSON", payload, file_name=f"{uploaded.name.rsplit('.', 1)[0]}.json",
                               mime="application/json", icon=":material/download:")
            st.json(res.data.model_dump(), expanded=2)
        else:
            st.caption("No JSON produced.")
