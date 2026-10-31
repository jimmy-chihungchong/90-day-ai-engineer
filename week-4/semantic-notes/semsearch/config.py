import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class Settings:
    provider: str
    base_url: str
    api_key: str
    chat_model: str
    embed_model: str
    timeout: float

    def client(self) -> OpenAI:
        return OpenAI(base_url=self.base_url, api_key=self.api_key or "lm-studio", timeout=self.timeout)


def get_settings(provider: str | None = None) -> Settings:
    provider = (provider or os.getenv("PROVIDER", "lmstudio")).lower()
    p = "NVIDIA" if provider == "nvidia" else "LMSTUDIO"
    defaults = {
        "NVIDIA": ("https://integrate.api.nvidia.com/v1", "openai/gpt-oss-20b", "nvidia/nv-embedqa-e5-v5"),
        "LMSTUDIO": ("http://localhost:1234/v1", "openai/gpt-oss-20b", "text-embedding-nomic-embed-text-v1.5"),
    }[p]
    return Settings(
        provider=provider,
        base_url=os.getenv(f"{p}_BASE_URL", defaults[0]),
        api_key=os.getenv(f"{p}_API_KEY", ""),
        chat_model=os.getenv(f"{p}_CHAT_MODEL", defaults[1]),
        embed_model=os.getenv(f"{p}_EMBED_MODEL", defaults[2]),
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
    )
