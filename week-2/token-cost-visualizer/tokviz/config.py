"""Settings loaded from .env (same local/nim switch as compare-chat)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Backend:
    name: str
    base_url: str
    api_key: str
    chat_model: str
    embed_model: str


def get_backend(name: str | None = None) -> Backend:
    name = (name or os.getenv("BACKEND", "local")).lower()
    if name == "nim":
        return Backend(
            "nim",
            os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            os.getenv("NIM_API_KEY", ""),
            os.getenv("NIM_CHAT_MODEL", "openai/gpt-oss-20b"),
            os.getenv("NIM_EMBED_MODEL", ""),
        )
    return Backend(
        "local",
        os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
        os.getenv("LMSTUDIO_CHAT_MODEL", "openai/gpt-oss-20b"),
        os.getenv("LMSTUDIO_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"),
    )


def timeout_seconds() -> float:
    return float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
