"""Сценарии из постановки задачи: дребезг у порога, короткоживущие процессы, неустранимый дисбаланс,
переменный интервал мониторинга. Случайные ряды строятся с фиксированным зерном."""

import numpy as np
import pytest

from trosna_gisteresis import hysteresis

SEED = 20261011
RULE = {"upper": 0.35, "lower": 0.25, "dwell": 30.0, "cooldown": 120.0}


def single_threshold(index, theta):
    """Постоянное окно: решение на каждой точке выше порога."""
    return int(np.sum(np.asarray(index) > theta))


def test_chattering_near_threshold_is_suppressed():
    rng = np.random.default_rng(SEED)
    t = np.arange(0.0, 3600.0, 5.0)
    index = 0.30 + 0.04 * rng.standard_normal(t.size)  # шум вокруг порога 0.30
    tr = hysteresis(index, t, **RULE)
    single = single_threshold(index, 0.30)
    assert single > 300
    assert tr.fired.sum() <= 3
    assert tr.fired.sum() * 100 < single


def test_short_lived_processes_are_ignored():
    t = np.arange(0.0, 1800.0, 1.0)
    index = np.full(t.size, 0.1)
    for start, length in [(100, 5), (400, 15), (700, 29), (1000, 10)]:
        index[start : start + length] = 0.6
    assert not hysteresis(index, t, **RULE).fired.any()
    index[1300:1360] = 0.6  # устойчивый дисбаланс 60 с
    tr = hysteresis(index, t, **RULE)
    # Первая точка выше порога — t = 1300, интервал до неё засчитывается: d = 30 при t = 1329.
    assert tr.fire_times.tolist() == [1329.0]


def test_unresolvable_imbalance_fires_once():
    # Ограничения не дают устранить дисбаланс: индекс час держится выше θ⁺.
    t = np.arange(0.0, 3600.0, 5.0)
    index = np.full(t.size, 0.5)
    tr = hysteresis(index, t, **RULE)
    assert tr.fire_times.tolist() == [30.0]
    assert single_threshold(index, RULE["upper"]) == t.size


def test_successful_rebalance_allows_next_decision():
    t = np.arange(0.0, 600.0, 5.0)
    index = np.where((t % 300) < 100, 0.5, 0.1)  # перераспределение помогает, затем дисбаланс возвращается
    tr = hysteresis(index, t, **RULE)
    # Второй раз выдержка считается от t = 295 (последняя точка ниже θ⁺, весь интервал засчитан).
    assert tr.fire_times.tolist() == [30.0, 325.0]
    assert hysteresis(index, t, **RULE, dwell_from="sample").fire_times.tolist() == [30.0, 330.0]


@pytest.mark.parametrize("seed", range(5))
def test_variable_monitoring_interval_delay_bounds(seed):
    rng = np.random.default_rng(SEED + seed)
    t = np.cumsum(rng.integers(1, 11, size=400)).astype(float)  # Δt от 1 до 10 с
    onset = 1000.0  # истинное начало дисбаланса
    index = np.where(t >= onset, 0.5, 0.1)
    first = t[t >= onset][0]
    for mode, lo, hi in [
        ("interval", RULE["dwell"] - 10, RULE["dwell"] + 10),
        ("sample", RULE["dwell"], RULE["dwell"] + 20),
    ]:
        tr = hysteresis(index, t, **RULE, dwell_from=mode)
        delay = tr.fire_times[0] - onset
        assert lo <= delay < hi, (mode, delay)
        if mode == "sample":
            assert tr.fire_times[0] - first >= RULE["dwell"]


def test_dwell_is_in_seconds_not_samples():
    for dt in (1.0, 2.0, 5.0, 10.0):
        t = np.arange(0.0, 200.0, dt)
        tr = hysteresis(np.full(t.size, 0.5), t, **RULE)
        assert tr.fire_times.tolist() == [30.0]
