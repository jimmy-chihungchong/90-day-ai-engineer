"""Split text into tokens using tiktoken encodings."""
from dataclasses import dataclass

import tiktoken

# Display name -> (tiktoken encoding, exact?)
# Only OpenAI-family encodings are exact; others are approximated with o200k_base.
ENCODINGS = {
    "o200k_base (gpt-oss / GPT-4o)": ("o200k_base", True),
    "cl100k_base (GPT-4 / GPT-3.5)": ("cl100k_base", True),
}


@dataclass(frozen=True)
class Token:
    id: int
    text: str    # decoded piece (invalid partial UTF-8 shown as U+FFFD)
    n_bytes: int


def tokenize(text: str, encoding: str = "o200k_base") -> list[Token]:
    enc = tiktoken.get_encoding(encoding)
    out = []
    for tid in enc.encode(text, disallowed_special=()):
        raw = enc.decode_single_token_bytes(tid)
        out.append(Token(tid, raw.decode("utf-8", errors="replace"), len(raw)))
    return out


def count_tokens(text: str, encoding: str = "o200k_base") -> int:
    return len(tiktoken.get_encoding(encoding).encode(text, disallowed_special=()))
