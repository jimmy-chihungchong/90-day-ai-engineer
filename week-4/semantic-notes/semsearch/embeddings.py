import hashlib
from pathlib import Path

import numpy as np

from .config import Settings
from .loaders import Chunk

BATCH = 16
MAX_EMBED_CHARS = 1800
CACHE_DIR = Path(".index")


class Embedder:
    def __init__(self, s: Settings):
        self.s, self.client = s, s.client()

    def _prep(self, texts: list[str], kind: str) -> list[str]:
        texts = [t[:MAX_EMBED_CHARS] for t in texts]
        if "nomic" in self.s.embed_model:  # nomic expects task prefixes
            prefix = "search_document: " if kind == "passage" else "search_query: "
            texts = [prefix + t for t in texts]
        return texts

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        extra = {"extra_body": {"input_type": kind, "truncate": "END"}} if self.s.provider == "nvidia" else {}
        vecs = []
        for i in range(0, len(texts), BATCH):
            r = self.client.embeddings.create(model=self.s.embed_model, input=self._prep(texts[i:i + BATCH], kind), **extra)
            vecs += [d.embedding for d in r.data]
        m = np.array(vecs, dtype=np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True).clip(min=1e-9)


class Index:
    """In-memory cosine index with an on-disk embedding cache keyed by chunk-text hash."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks, self.vectors = chunks, vectors

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, progress=None) -> "Index":
        CACHE_DIR.mkdir(exist_ok=True)
        cache_file = CACHE_DIR / (hashlib.md5(embedder.s.embed_model.encode()).hexdigest()[:10] + ".npz")
        cache: dict[str, np.ndarray] = {}
        if cache_file.exists():
            z = np.load(cache_file, allow_pickle=False)
            cache = dict(zip(z["keys"].tolist(), z["vecs"]))
        keys = [hashlib.md5(c.embed_text.encode()).hexdigest() for c in chunks]
        missing = [i for i, k in enumerate(keys) if k not in cache]
        for start in range(0, len(missing), 64):
            idx = missing[start:start + 64]
            for i, v in zip(idx, embedder.embed([chunks[i].embed_text for i in idx])):
                cache[keys[i]] = v
            if progress:
                progress(min(start + 64, len(missing)) / len(missing))
        if missing:
            np.savez(cache_file, keys=np.array(list(cache)), vecs=np.array(list(cache.values())))
        return cls(chunks, np.array([cache[k] for k in keys]))

    def search(self, query_vec: np.ndarray, k: int = 8) -> list[tuple[float, Chunk]]:
        scores = self.vectors @ query_vec
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in top]
