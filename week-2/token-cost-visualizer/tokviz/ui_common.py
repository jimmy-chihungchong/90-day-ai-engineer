"""Shared Streamlit bits for the LLM-backed tabs: model picker and the slow-model warning."""
import streamlit as st

from .config import get_backend, timeout_seconds
from .llm import LLMError, LLMTimeout, list_models

PREFERRED = "llama-3.2-3b-instruct"


@st.cache_data(ttl=30, show_spinner=False)
def _models(backend_name: str) -> list[str]:
    return [m for m in list_models(get_backend(backend_name)) if "embed" not in m.lower()]


def _use_model(model_key: str, name: str):
    st.session_state[model_key] = name
    st.session_state.pop(f"{model_key}_timeout", None)


def model_picker(key: str):
    """Render a model selectbox. Returns (backend, model, available_models) or None if unusable."""
    backend = get_backend()
    try:
        models = _models(backend.name)
    except LLMError as e:
        st.error(str(e), icon=":material/error:")
        return None
    if not models:
        st.warning("No chat models found on the server.")
        return None
    if key not in st.session_state:
        st.session_state[key] = next((m for m in models if PREFERRED in m), models[0])
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
