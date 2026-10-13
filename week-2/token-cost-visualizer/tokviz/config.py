"""Settings loaded from .env (same local/nim switch as compare-chat)."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# NIM lists ~80 models but most return 404. These were verified working with a real key on 2026-10-13.
# The default is a fast NON-reasoning model: reasoning models spend the small token budgets used by
# the tabs on hidden thinking, so they often return an empty answer.
NIM_DEFAULT_CHAT = "nvidia/ising-calibration-1.5-31b"
NIM_VERIFIED_CHAT = (
    NIM_DEFAULT_CHAT,                                   # fast, no reasoning, logprobs (a qubit-calibration specialist)
    "nvidia/nemotron-3-super-120b-a12b",                # fast, but reasoning
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",    # fast, reasoning, sometimes 503 (capacity)
    "openai/gpt-oss-20b",                               # slow (10-18s), reasoning
)
NIM_DEFAULT_EMBED = "nvidia/nemotron-3-embed-1b"


@dataclass(frozen=True)
class Backend:
    name: str
    base_url: str
    api_key: str
    chat_model: str
    embed_model: str
    chat_models: tuple[str, ...] = ()   # curated choices; empty = ask the server what is loaded


def _csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(m.strip() for m in raw.split(",") if m.strip()) or default


def get_backend(name: str | None = None) -> Backend:
    name = (name or os.getenv("BACKEND", "local")).lower()
    if name == "nim":
        return Backend(
            "nim",
            os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            os.getenv("NIM_API_KEY", ""),
            os.getenv("NIM_CHAT_MODEL", NIM_DEFAULT_CHAT),
            os.getenv("NIM_EMBED_MODEL", NIM_DEFAULT_EMBED),
            _csv("NIM_CHAT_MODELS", NIM_VERIFIED_CHAT),
        )
    return Backend(
        "local",
        os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
        os.getenv("LMSTUDIO_CHAT_MODEL", "openai/gpt-oss-20b"),
        os.getenv("LMSTUDIO_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"),
    )


def backend_options() -> list[str]:
    """Backends to offer in the UI. A cloud deploy (BACKEND=nim) hides LM Studio: it isn't reachable there."""
    return ["nim"] if os.getenv("BACKEND", "local").lower() == "nim" else ["local", "nim"]


def timeout_seconds() -> float:
    return float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
