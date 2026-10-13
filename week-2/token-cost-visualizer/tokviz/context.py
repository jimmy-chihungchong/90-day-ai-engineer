"""What fits in a context window: split a token list into what the model keeps and what it loses."""
from dataclasses import dataclass

from .tokenizer import Token


@dataclass(frozen=True)
class WindowSplit:
    dropped_start: list[Token]   # cut off the front (when keeping the most recent)
    kept: list[Token]
    dropped_end: list[Token]     # cut off the back (when keeping the beginning)
    budget: int                  # tokens available for input = window - reserved output

    @property
    def n_dropped(self) -> int:
        return len(self.dropped_start) + len(self.dropped_end)


def split_window(tokens: list[Token], window: int, reserve_output: int = 0,
                 keep: str = "recent") -> WindowSplit:
    """Input and output share one window, so input gets `window - reserve_output` tokens.

    keep="recent": drop the oldest tokens (what most chat apps do).
    keep="start":  drop the newest tokens.
    """
    if keep not in ("recent", "start"):
        raise ValueError("keep must be 'recent' or 'start'")
    budget = max(window - max(reserve_output, 0), 0)
    if len(tokens) <= budget:
        return WindowSplit([], list(tokens), [], budget)
    if keep == "recent":
        cut = len(tokens) - budget
        return WindowSplit(tokens[:cut], tokens[cut:], [], budget)
    return WindowSplit([], tokens[:budget], tokens[budget:], budget)


def text_of(tokens: list[Token]) -> str:
    return "".join(t.text for t in tokens)
