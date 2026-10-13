"""Shared Streamlit bits for the LLM-backed tabs: model picker and the slow-model warning."""
import streamlit as st

from .config import backend_options, get_backend, timeout_seconds
from .llm import LLMError, LLMTimeout, list_models

PREFERRED = "llama-3.2-3b-instruct"
BACKEND_KEY = "backend"
BACKEND_LABELS = {"local": "Local (LM Studio)", "nim": "Online (NVIDIA NIM)"}


def current_backend():
    """The backend chosen in the sidebar (falls back to the BACKEND env setting)."""
    return get_backend(st.session_state.get(BACKEND_KEY))


def backend_selector():
    """Sidebar switch between LM Studio and NIM. Hides LM Studio on a cloud deploy (BACKEND=nim)."""
    options = backend_options()
    if st.session_state.get(BACKEND_KEY) not in options:
        st.session_state[BACKEND_KEY] = options[0] if len(options) == 1 else get_backend().name
    with st.sidebar:
        if len(options) > 1:
            st.radio("Model server", options, key=BACKEND_KEY, format_func=BACKEND_LABELS.get,
                     help="Used by the Temperature, Hallucination and Embeddings tabs.")
        else:
            st.caption(f"Model server: {BACKEND_LABELS[options[0]]}")
        b = get_backend(st.session_state[BACKEND_KEY])
        if b.name == "nim" and not b.api_key:
            st.warning("No NIM_API_KEY set. Add it to .env (or the app's Secrets).", icon=":material/key:")


@st.cache_data(ttl=30, show_spinner=False)
def _models(backend_name: str) -> list[str]:
    return [m for m in list_models(get_backend(backend_name)) if "embed" not in m.lower()]


def _use_model(model_key: str, name: str):
    st.session_state[model_key] = name
    st.session_state.pop(f"{model_key}_timeout", None)


def model_picker(key: str):
    """Render a model selectbox. Returns (backend, model, available_models) or None if unusable."""
    backend = current_backend()
    if backend.name == "nim" and not backend.api_key:
        st.warning("Add NIM_API_KEY to .env (or the app's Secrets) to use the online models.",
                   icon=":material/key:")
        return None
    if backend.chat_models:     # curated list (NIM lists many models that 404)
        models = list(backend.chat_models)
    else:
        try:
            models = _models(backend.name)
        except LLMError as e:
            st.error(str(e), icon=":material/error:")
            return None
    if not models:
        st.warning("No chat models found on the server.")
        return None
    if st.session_state.get(key) not in models:     # first run, or the backend was switched
        st.session_state[key] = next((m for m in models if PREFERRED in m or m == backend.chat_model), models[0])
    st.selectbox("Model", models, key=key, help="Smaller models answer faster.")
    return backend, st.session_state[key], models


def record_timeout(key: str, e: LLMTimeout):
    st.session_state[f"{key}_timeout"] = (e.model, e.suggestion)


def clear_timeout(key: str):
    st.session_state.pop(f"{key}_timeout", None)


def show_timeout(key: str):
    """If the last call timed out, warn and offer one click to switch to the suggested model."""
    if f"{key}_timeout" not in st.session_state:
        return
    slow, alt = st.session_state[f"{key}_timeout"]
    st.warning(f"**{slow}** didn't answer within {timeout_seconds():.0f}s.", icon=":material/hourglass_top:")
    if alt:
        st.button(f"Switch to {alt}", on_click=_use_model, args=(key, alt),
                  icon=":material/swap_horiz:", key=f"{key}_switch")
