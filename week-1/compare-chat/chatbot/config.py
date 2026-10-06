"""Module 1: load .env, expose backend + model registry."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Backend:
    name: str
    base_url: str
    api_key: str
    models: tuple  # (model_a_id, model_b_id)


def _env(key, default=""):
    """Env var first, then Streamlit secrets (for deployment)."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        return str(st.secrets.get(key, default))
    except Exception:
        return default


def get_backend(name=None):
    name = (name or _env("BACKEND", "local")).lower()
    if name == "nim":
        return Backend(
            "nim",
            _env("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            _env("NIM_API_KEY"),
            (_env("NIM_MODEL_A", "openai/gpt-oss-20b"),
             _env("NIM_MODEL_B", "meta/muse-glimmer-30b")),
        )
    return Backend(
        "local",
        _env("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        _env("LMSTUDIO_API_KEY", "lm-studio"),
        (_env("LMSTUDIO_MODEL_A", "microsoft/phi-4-mini-reasoning"),
         _env("LMSTUDIO_MODEL_B", "llama-3.2-3b-instruct")),
    )


def get_timeout():
    return float(_env("LLM_TIMEOUT_SECONDS", "45"))
