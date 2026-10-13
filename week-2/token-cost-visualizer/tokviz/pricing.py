"""Cost estimates per model from data/pricing.json."""
import json
from dataclasses import dataclass
from pathlib import Path

from .tokenizer import count_tokens

PRICING_FILE = Path(__file__).resolve().parent.parent / "data" / "pricing.json"


@dataclass(frozen=True)
class ModelPrice:
    id: str
    label: str
    kind: str                 # "local" | "paid"
    input: float              # USD per 1M tokens
    output: float
    context_window: int | None
    encoding: str
    exact: bool


def load_models(path: Path = PRICING_FILE) -> tuple[list[ModelPrice], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ModelPrice(**m) for m in data["models"]], data.get("_as_of", "unknown")


def cost_usd(tokens: int, per_million: float) -> float:
    return tokens * per_million / 1_000_000


def estimate(text: str, expected_output_tokens: int, models: list[ModelPrice]) -> list[dict]:
    """One row per model: token count, input/output/total cost."""
    rows = []
    for m in models:
        n = count_tokens(text, m.encoding)
        cin, cout = cost_usd(n, m.input), cost_usd(expected_output_tokens, m.output)
        rows.append({
            "model": m.label, "tokens": n, "exact": m.exact,
            "input_cost": cin, "output_cost": cout, "total": cin + cout,
            "context_used_pct": (100 * n / m.context_window) if m.context_window else None,
        })
    return rows
