"""Pure sampling math: how temperature and top_p reshape a next-token distribution."""
import math


def to_probs(logprobs: list[float]) -> list[float]:
    """Convert logprobs to probabilities renormalized over the candidates given (top-k only)."""
    ps = [math.exp(lp) for lp in logprobs]
    total = sum(ps)
    return [p / total for p in ps]


def apply_temperature(probs: list[float], temperature: float) -> list[float]:
    """p_i ** (1/T), renormalized. T=0 means greedy (all mass on the most likely token)."""
    if not probs:
        return []
    if temperature <= 0:
        best = max(range(len(probs)), key=probs.__getitem__)
        return [1.0 if i == best else 0.0 for i in range(len(probs))]
    scaled = [p ** (1.0 / temperature) for p in probs]
    total = sum(scaled)
    return [s / total for s in scaled]


def apply_top_p(probs: list[float], top_p: float) -> list[float]:
    """Keep the smallest set of top tokens whose cumulative probability reaches top_p; renormalize."""
    if not probs or top_p >= 1.0:
        return list(probs)
    order = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
    keep, cum = set(), 0.0
    for i in order:
        keep.add(i)
        cum += probs[i]
        if cum >= top_p:
            break
    kept_total = sum(probs[i] for i in keep)
    return [probs[i] / kept_total if i in keep else 0.0 for i in range(len(probs))]


def reshape(probs: list[float], temperature: float, top_p: float = 1.0) -> list[float]:
    """Order used by most servers: temperature first, then top_p."""
    return apply_top_p(apply_temperature(probs, temperature), top_p)
