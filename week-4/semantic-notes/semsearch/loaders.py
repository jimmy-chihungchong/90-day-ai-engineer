"""Turn .md / .txt / .json files into small searchable chunks."""
import json
import re
from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 1200
SUPPORTED = {".md", ".txt", ".json"}


@dataclass
class Chunk:
    source: str  # file name
    title: str  # nearest heading / record title
    text: str

    @property
    def embed_text(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text


def _pack(paragraphs: list[str], limit: int = MAX_CHARS) -> list[str]:
    """Greedily pack paragraphs into <=limit pieces, hard-splitting oversized ones."""
    out, cur = [], ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        while len(p) > limit:
            cut = p.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:cut])
            p = p[cut:].strip()
        if cur and len(cur) + len(p) + 2 > limit:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out


def _markdown(name: str, text: str) -> list[Chunk]:
    sections, title, buf = [], Path(name).stem, []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            sections.append((title, "\n".join(buf)))
            title, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    sections.append((title, "\n".join(buf)))
    return [
        Chunk(name, f"{Path(name).stem} > {t}", piece)
        for t, body in sections
        for piece in _pack(re.split(r"\n\s*\n", body))
    ]


def _plain(name: str, text: str) -> list[Chunk]:
    return [Chunk(name, Path(name).stem, p) for p in _pack(re.split(r"\n\s*\n", text))]


def _record_text(rec) -> str:
    if isinstance(rec, dict):
        return "\n".join(
            f"{k}: {json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v}" for k, v in rec.items()
        )
    return str(rec)


def _record_title(rec, fallback: str) -> str:
    if isinstance(rec, dict):
        for k in ("title", "name", "subject", "date", "job_id", "id", "url"):
            if rec.get(k):
                return f"{fallback} > {rec[k]}"
    return fallback


def _json(name: str, text: str) -> list[Chunk]:
    data = json.loads(text)
    stem = Path(name).stem
    records: list[tuple[str, object]] = []
    if isinstance(data, list):
        records = [(stem, r) for r in data]
    elif isinstance(data, dict):
        scalars = {k: v for k, v in data.items() if not isinstance(v, list)}
        if scalars:
            records.append((stem, scalars))
        for k, v in data.items():
            if isinstance(v, list):
                records += [(f"{stem}.{k}", r) for r in v]
    else:
        records = [(stem, data)]
    chunks = []
    for base, rec in records:
        title = _record_title(rec, base)
        chunks += [Chunk(name, title, piece) for piece in _pack([_record_text(rec)])]
    return chunks


def load_text(name: str, text: str) -> list[Chunk]:
    ext = Path(name).suffix.lower()
    if ext == ".md":
        return _markdown(name, text)
    if ext == ".json":
        try:
            return _json(name, text)
        except json.JSONDecodeError:
            return _plain(name, text)
    return _plain(name, text)


def load_folder(folder: str) -> list[Chunk]:
    chunks = []
    for f in sorted(Path(folder).rglob("*")):
        if f.suffix.lower() in SUPPORTED and f.is_file():
            chunks += load_text(f.name, f.read_text(encoding="utf-8", errors="replace"))
    return chunks
