"""Embeddings: fetch vectors, compare by cosine similarity, squash to 2D for plotting (numpy only)."""
import numpy as np
import openai

from .config import Backend
from .llm import LLMError, LLMTimeout, backend_models, make_client, suggest_alternative


def embed(backend: Backend, texts: list[str], model: str | None = None, timeout: float = 45.0,
          client=None, kind: str = "passage") -> np.ndarray:
    """Return an (n, dim) array of L2-normalized vectors. Raises LLMTimeout / LLMError.

    kind: "passage" for the things you search over, "query" for the thing you search with.
    NIM's embedding models need this hint (without it, similar sentences score no closer than
    unrelated ones). LM Studio's nomic model gets no hint.
    """
    model = model or backend.embed_model
    if not model:
        raise LLMError(f"No embedding model configured for the {backend.name} backend "
                       f"(set {'NIM_EMBED_MODEL' if backend.name == 'nim' else 'LMSTUDIO_EMBED_MODEL'}).")
    client = client or make_client(backend, timeout)
    extra = {"extra_body": {"input_type": kind}} if backend.name == "nim" else {}
    try:
        r = client.embeddings.create(model=model, input=texts, **extra)
    except openai.APITimeoutError:
        # NIM lists many embedders that 404, so only suggest alternatives from a local server's list
        alt = suggest_embedder(model, backend_models(backend)) if backend.name == "local" else None
        raise LLMTimeout(model, alt)
    except openai.APIConnectionError as e:
        raise LLMError(f"Cannot reach {backend.base_url} ({backend.name}). Is the server running? {e}")
    except openai.APIStatusError as e:
        raise LLMError(f"{backend.name} returned {e.status_code}: {e.message}")
    return normalize(np.array([d.embedding for d in r.data], dtype=float))


def suggest_embedder(current: str, available: list[str]) -> str | None:
    others = [m for m in available if "embed" in m.lower() and m != current]
    return others[0] if others else None


def normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norms == 0, 1, norms)


def cosine_matrix(v: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of already-normalized vectors."""
    return v @ v.T


def rank_by_similarity(query: np.ndarray, docs: np.ndarray) -> list[tuple[int, float]]:
    """(index, cosine) for each doc, best first. Vectors must be normalized."""
    sims = docs @ query
    return [(int(i), float(sims[i])) for i in np.argsort(-sims)]


def project_2d(v: np.ndarray) -> np.ndarray:
    """PCA to 2 dimensions via SVD. Needs >= 2 rows; pads with zeros if rank < 2."""
    if len(v) < 2:
        raise ValueError("Need at least 2 texts to draw a map.")
    centered = v - v.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ vt[:2].T
    if xy.shape[1] < 2:
        xy = np.hstack([xy, np.zeros((len(xy), 2 - xy.shape[1]))])
    return xy
