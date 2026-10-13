import httpx
import numpy as np
import openai
import pytest

from tokviz.config import Backend, get_backend
from tokviz.embeddings import (cosine_matrix, embed, normalize, project_2d,
                               rank_by_similarity, suggest_embedder)
from tokviz.llm import LLMError, LLMTimeout, list_models

REQ = httpx.Request("POST", "http://x/v1/embeddings")
B = get_backend("local")


class FakeClient:
    def __init__(self, exc=None, vectors=None):
        self.embeddings = self
        self._exc, self._vectors = exc, vectors

    def create(self, **kw):
        if self._exc:
            raise self._exc

        class D:  # mimics response.data[i].embedding
            def __init__(s, e): s.embedding = e
        return type("R", (), {"data": [D(v) for v in self._vectors]})


def test_normalize_unit_length_and_zero_safe():
    v = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.linalg.norm(v[0]) == pytest.approx(1) and not np.isnan(v).any()


def test_cosine_identity_and_orthogonal():
    m = cosine_matrix(normalize(np.array([[1.0, 0], [0, 1.0], [2.0, 0]])))
    assert m[0, 2] == pytest.approx(1) and m[0, 1] == pytest.approx(0)


def test_rank_by_similarity_orders_best_first():
    docs = normalize(np.array([[0.0, 1], [1, 0.1], [1, 1]]))
    q = normalize(np.array([[1.0, 0]]))[0]
    assert [i for i, _ in rank_by_similarity(q, docs)] == [1, 2, 0]


def test_project_2d_shape_and_preserves_cluster_gap():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.05, (5, 20)) + np.eye(20)[0]
    b = rng.normal(0, 0.05, (5, 20)) + np.eye(20)[1]
    xy = project_2d(normalize(np.vstack([a, b])))
    assert xy.shape == (10, 2)
    within = np.linalg.norm(xy[:5] - xy[:5].mean(0), axis=1).max()
    between = np.linalg.norm(xy[:5].mean(0) - xy[5:].mean(0))
    assert between > 3 * within


def test_project_2d_needs_two_rows():
    with pytest.raises(ValueError):
        project_2d(np.ones((1, 4)))


def test_embed_normalizes_with_fake_client():
    v = embed(B, ["a", "b"], model="m", client=FakeClient(vectors=[[3, 4], [0, 2]]))
    assert np.allclose(np.linalg.norm(v, axis=1), 1)


def test_embed_errors_are_friendly():
    with pytest.raises(LLMTimeout):
        embed(B, ["a"], model="m", client=FakeClient(exc=openai.APITimeoutError(request=REQ)))
    with pytest.raises(LLMError, match="Is the server running"):
        embed(B, ["a"], model="m", client=FakeClient(exc=openai.APIConnectionError(request=REQ)))
    with pytest.raises(LLMError, match="No embedding model"):
        embed(Backend("nim", "http://x", "k", "chat", ""), ["a"], client=FakeClient(vectors=[[1.0]]))


def test_suggest_embedder():
    assert suggest_embedder("a-embed", ["a-embed", "b-embed", "chat"]) == "b-embed"
    assert suggest_embedder("a-embed", ["a-embed", "chat"]) is None


def _live():
    try:
        return list_models(B, timeout=2)
    except LLMError:
        return None


@pytest.mark.skipif(_live() is None, reason="LM Studio not running")
def test_live_similar_sentences_score_higher():
    v = embed(B, ["The cat sat on the sofa.", "A kitten is sleeping on the couch.",
                  "The stock market fell sharply today."], timeout=90)
    assert v.shape[0] == 3 and v.shape[1] > 100
    m = cosine_matrix(v)
    assert m[0, 1] > m[0, 2] + 0.1


def test_parse_lines():
    from tokviz.tab_embeddings import parse_lines
    rows = parse_lines("animals | cat\n\n  money|stocks  \nplain line\n | no topic\nx |")
    assert rows == [("animals", "cat"), ("money", "stocks"), ("other", "plain line"),
                    ("other", "no topic"), ("other", "x |")]
