"""Turn per-token logprobs into confidence bands and a short summary."""
import math
from dataclasses import dataclass

from .llm import TokenLP

SHAKY = 0.5    # below this a token was a coin-flip or worse
SURE = 0.9     # at or above this the model was very sure


def prob(t: TokenLP) -> float:
    return math.exp(t.logprob)


def band(p: float) -> str:
    """'sure' | 'unsure' | 'shaky'"""
    return "sure" if p >= SURE else "unsure" if p >= SHAKY else "shaky"


@dataclass(frozen=True)
class Summary:
    mean: float
    lowest_token: str
    lowest_p: float
    n_shaky: int
    n_tokens: int


def summarize(tokens: list[TokenLP]) -> Summary | None:
    if not tokens:
        return None
    ps = [prob(t) for t in tokens]
    i = min(range(len(ps)), key=ps.__getitem__)
    return Summary(sum(ps) / len(ps), tokens[i].token, ps[i],
                   sum(p < SHAKY for p in ps), len(ps))
