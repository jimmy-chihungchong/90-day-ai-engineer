import openai

from .config import Settings
from .loaders import Chunk

SYSTEM = (
    "You answer questions about the user's own notes. Use ONLY the numbered excerpts provided. "
    "Cite sources like [1], [2]. If the excerpts do not contain the answer, say so briefly. Be concise."
)
CONTEXT_BUDGET = 9000  # chars; keeps small local context windows (e.g. 8k tokens) safe


class LLMTimeout(Exception):
    pass


def suggest_alternatives(s: Settings) -> list[str]:
    """Other chat models on the same endpoint (embedding models excluded)."""
    try:
        ids = [m.id for m in s.client().models.list().data]
    except Exception:
        ids = ["meta/llama-3.3-70b-instruct", "qwen/qwen3-next-80b-a3b-instruct"] if s.provider == "nvidia" else []
    return [i for i in ids if i != s.chat_model and "embed" not in i.lower()][:4]


def answer(s: Settings, question: str, hits: list[tuple[float, Chunk]], model: str | None = None) -> str:
    parts, used = [], 0
    for n, (_, c) in enumerate(hits, 1):
        block = f"[{n}] ({c.source} | {c.title})\n{c.text}"
        if used + len(block) > CONTEXT_BUDGET:
            break
        parts.append(block)
        used += len(block)
    try:
        r = s.client().chat.completions.create(
            model=model or s.chat_model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Excerpts:\n\n" + "\n\n".join(parts) + f"\n\nQuestion: {question}"},
            ],
            max_tokens=2000,
            temperature=0.2,
        )
    except (openai.APITimeoutError, openai.APIConnectionError) as e:
        raise LLMTimeout(str(e)) from e
    return (r.choices[0].message.content or "").strip() or "(model returned no text; it may have spent its token budget reasoning)"
