import math

import pytest

from tokviz.sampling import apply_temperature, apply_top_p, reshape, to_probs

P = [0.6, 0.3, 0.1]


def test_to_probs_sums_to_one():
    assert sum(to_probs([math.log(0.5), math.log(0.2)])) == pytest.approx(1)


def test_temperature_zero_is_greedy():
    assert apply_temperature(P, 0) == [1.0, 0.0, 0.0]


def test_temperature_one_is_identity():
    assert apply_temperature(P, 1.0) == pytest.approx(P)


def test_low_temp_sharpens_high_temp_flattens():
    assert apply_temperature(P, 0.5)[0] > P[0]
    assert apply_temperature(P, 2.0)[0] < P[0]
    assert apply_temperature(P, 100)[2] == pytest.approx(1 / 3, abs=0.02)


def test_all_outputs_sum_to_one_and_keep_order():
    for t in (0.2, 0.7, 1.5, 3):
        out = apply_temperature(P, t)
        assert sum(out) == pytest.approx(1) and out[0] >= out[1] >= out[2]


def test_top_p_cuts_tail_and_renormalizes():
    out = apply_top_p(P, 0.8)       # 0.6 + 0.3 >= 0.8 -> drop 0.1
    assert out[2] == 0 and sum(out) == pytest.approx(1)
    assert out[0] == pytest.approx(0.6 / 0.9)


def test_top_p_always_keeps_best_and_one_means_all():
    assert apply_top_p(P, 0.01) == [1.0, 0.0, 0.0]
    assert apply_top_p(P, 1.0) == P


def test_reshape_and_empty():
    assert reshape(P, 0, 0.5) == [1.0, 0.0, 0.0]
    assert apply_temperature([], 1) == [] and apply_top_p([], 0.5) == []
