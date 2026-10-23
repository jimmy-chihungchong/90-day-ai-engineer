"""Check which NVIDIA NIM models are callable with your key, and how fast they answer.

Usage:  python scripts/check_nim.py [model ...]     (needs NIM_API_KEY in env or .env)
Without model args it tests openai/gpt-oss-20b plus the fallbacks in config.py.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resume_extractor.config import NIM_FALLBACKS, get_settings  # noqa: E402
from resume_extractor.llm import LLMClient, LLMError, extract_json  # noqa: E402

PROMPT = 'Reply with only this JSON: {"name": "Ada", "skills": ["Python"]}'


def main():
    base = get_settings({"LLM_BACKEND": "nim"})
    client = LLMClient(base)
    catalog = set(client.list_models())
    print(f"catalog: {len(catalog)} models\n")
    for model in sys.argv[1:] or [base.model, *NIM_FALLBACKS]:
        if model not in catalog:
            print(f"MISSING  {model} (not in catalog)")
            continue
        t = time.perf_counter()
        try:
            out = client.complete("Output only JSON.", PROMPT, model=model, max_tokens=1500)
            ok = "OK" if extract_json(out).get("name") == "Ada" else "BADJSON"
            print(f"{ok:8s} {model}  {time.perf_counter() - t:.1f}s")
        except (LLMError, ValueError) as e:
            print(f"ERROR    {model}  {str(e)[:120]}")


if __name__ == "__main__":
    main()
