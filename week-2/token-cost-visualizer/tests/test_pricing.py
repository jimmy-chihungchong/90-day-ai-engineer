import pytest

from tokviz.pricing import cost_usd, estimate, load_models


def test_cost_math():
    assert cost_usd(1_000_000, 2.5) == pytest.approx(2.5)
    assert cost_usd(0, 10) == 0


def test_load_models_valid():
    models, as_of = load_models()
    assert len(models) >= 5 and as_of != "UNVERIFIED"
    assert all(m.input >= 0 and m.output >= 0 for m in models)
    assert len({m.id for m in models}) == len(models)


def test_estimate_local_free_paid_not():
    models, _ = load_models()
    rows = estimate("Hello world " * 100, 500, models)
    assert len(rows) == len(models)
    for m, r in zip(models, rows):
        assert r["tokens"] > 0
        assert (r["total"] == 0) == (m.kind == "local")
    assert rows[0]["context_used_pct"] is not None
