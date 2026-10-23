"""Backend selection and settings, loaded from .env / Streamlit secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from dotenv import load_dotenv

load_dotenv()

# Fallbacks for NIM. Catalog presence is NOT enough (many listed models return 404 when called),
# so these were each verified with a real chat call on a free-tier key on 2026-10-23
# (scripts/check_nim.py). Re-check before relying on them.
NIM_FALLBACKS = [
    "minimaxai/minimax-m3.1-flash",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "z-ai/glm-5.3",
]


@dataclass(frozen=True)
class Settings:
    backend: str  # "lmstudio" | "nim"
    base_url: str
    model: str
    api_key: str
    timeout: float


def get_settings(overrides: Mapping[str, str] | None = None) -> Settings:
    """Read settings from env; `overrides` (e.g. st.secrets) wins over env vars."""
    def env(key: str, default: str = "") -> str:
        if overrides and overrides.get(key):
            return str(overrides[key])
        return os.getenv(key, default)

    backend = env("LLM_BACKEND", "lmstudio").lower()
    if backend == "nim":
        base_url = env("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        model = env("NIM_MODEL", "openai/gpt-oss-20b")
        api_key = env("NIM_API_KEY")
        if not api_key:
            raise ValueError("NIM_API_KEY is not set (LLM_BACKEND=nim).")
    elif backend == "lmstudio":
        base_url = env("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        model = env("LMSTUDIO_MODEL", "openai/gpt-oss-20b")
        api_key = env("LMSTUDIO_API_KEY", "lm-studio")
    else:
        raise ValueError(f"Unknown LLM_BACKEND '{backend}'. Use 'lmstudio' or 'nim'.")
    return Settings(backend, base_url, model, api_key, float(env("LLM_TIMEOUT_SECONDS", "90")))
