"""Turn tokens into Streamlit-markdown 'chips' (alternating background colors)."""
import re

from .tokenizer import Token

COLORS = ["blue", "green", "orange", "violet", "red", "yellow"]
_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|~<>$:&^=@%])")
MAX_CHIPS = 1500  # keep the browser snappy on huge pastes


def escape_md(text: str) -> str:
    """Escape markdown syntax so arbitrary token text renders literally."""
    return _SPECIAL.sub(lambda m: "\\" + m.group(1), text)


def visible(text: str) -> str:
    """Make whitespace visible and escape markdown syntax."""
    return escape_md(text.replace(" ", "·").replace("\n", "↵").replace("\t", "→"))


def chips_markdown(tokens: list[Token]) -> str:
    parts = [
        f":{COLORS[i % len(COLORS)]}-background[{visible(t.text)}]"
        for i, t in enumerate(tokens[:MAX_CHIPS])
    ]
    out = " ".join(parts)
    if len(tokens) > MAX_CHIPS:
        out += f"\n\n*…{len(tokens) - MAX_CHIPS:,} more tokens not shown*"
    return out
