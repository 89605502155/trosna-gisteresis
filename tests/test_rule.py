"""Поведение правила (9)–(11) на коротких рядах, разобранных вручную."""

import math

import numpy as np
import pytest

from trosna_gisteresis import Hysteresis, HysteresisState, hysteresis

NAN = math.nan
RULE = {"upper": 2.0, "lower": 1.0, "dwell": 3.0, "cooldown": 10.0}


def fires(index, t=None, **changes):
    return hysteresis(index, t, **{**RULE, **changes}).fired.tolist()


def test_dwell_counts_interval_before_first_exceedance():
    # t = 0 — первая точка, Δt не определён; d = 0, 1, 2, 3 → срабатывание на t = 3.
    assert fires([3, 3, 3, 3, 3]) == [False, False, False, True, False]
    # Первая точка ниже порога: превышение с t = 1 засчитывается с t = 0 (весь интервал Δt).
    assert fires([0, 3, 3, 3, 3]) == [False, False, False, True, False]


def test_dwell_from_sample_counts_from_first_exceeding_point():
    assert fires([0, 3, 3, 3, 3], dwell_from="sample") == [False, False, False, False, True]


def test_dwell_must_be_continuous():
    # I = θ⁺ прерывает серию (неравенство строгое); новый отсчёт идёт от точки разрыва t = 3.
    assert not any(fires([3, 3, 3, 2, 3, 3]))
    assert fires([3, 3, 3, 2, 3, 3, 3]) == [False] * 6 + [True]


def test_thresholds_are_strict():
    # I = θ⁻ не взводит правило, I чуть ниже θ⁻ — взводит.
    after_fire = [3, 3, 3, 3]
    gap = [1.0] * 10
    again = [3, 3, 3, 3]
    assert sum(fires(after_fire + gap + again)) == 1
    assert sum(fires([*after_fire, 0.999, *gap[1:], *again])) == 2


def test_rearm_requires_drop_below_lower():
    # Индекс опускается между порогами — правило не взводится, повторного срабатывания нет.
    series = [3] * 4 + [1.5] * 20 + [3] * 10
    assert sum(fires(series)) == 1
    series = [3] * 4 + [0.5] + [1.5] * 19 + [3] * 10
    assert sum(fires(series)) == 2


def test_cooldown_blocks_early_second_fire():
    series = [3] * 4 + [0] + [3] * 4  # второе срабатывание готово на t = 8, но t − τ = 5 < 10
    out = fires(series)
    assert out.index(True) == 3
    assert sum(out) == 1
    out = fires([*series, 3, 3, 3, 3, 3, 3])  # дождались t = 13: t − τ = 10 ≥ T_c
    assert [k for k, f in enumerate(out) if f] == [3, 13]


def test_cooldown_zero_and_dwell_zero():
    # T_d = 0: срабатывание на первой же точке выше θ⁺, но не ниже него.
    assert fires([0, 1.5, 3, 0, 3], dwell=0, cooldown=0) == [False, False, True, False, True]


def test_initially_disarmed_requires_drop_first():
    assert not any(fires([3] * 10, armed=False))
    assert fires([0, 3, 3, 3, 3], armed=False) == [False, False, False, True, False]


def test_variable_monitoring_interval_dwell_in_time_units():
    t = [0, 10, 20, 21, 22]
    out = hysteresis([3] * 5, t, upper=2, lower=1, dwell=21, cooldown=0)
    assert out.fired.tolist() == [False, False, False, True, False]
    assert out.dwell.tolist() == [0, 10, 20, 21, 1]


def test_missing_reset_breaks_dwell():
    assert not any(fires([3, 3, NAN, 3, 3]))
    assert fires([3, 3, NAN, 3, 3, 3]) == [False] * 5 + [True]


def test_missing_hold_keeps_dwell():
    assert fires([3, 3, NAN, 3, 3], missing="hold") == [False, False, False, True, False]
    # Пропуск никогда не даёт срабатывания сам по себе, даже если выдержка уже набрана.
    assert fires([3, 3, 3, NAN], dwell=3.5, missing="hold") == [False] * 4


def test_missing_never_arms():
    assert not any(fires([NAN] * 3 + [3] * 5, armed=False))


def test_step_reports_values_checked_by_rule():
    h = Hysteresis(**RULE)
    steps = [h.step(3, t) for t in range(5)]
    fired = steps[3]
    assert (fired.fired, fired.dwell, fired.armed, fired.since_fired) == (True, 3.0, True, math.inf)
    after = steps[4]
    assert (after.fired, after.dwell, after.armed, after.since_fired) == (False, 1.0, False, 1.0)
    assert h.state == HysteresisState(t=4.0, run_start=3.0, armed=False, last_fired=3.0)


def test_time_must_increase_and_be_finite():
    h = Hysteresis(**RULE)
    h.step(0, 5)
    for bad in (5, 4, math.nan, math.inf):
        with pytest.raises(ValueError, match="время"):
            h.step(0, bad)
    assert h.state.t == 5.0  # ошибка не портит состояние


def test_update_continues_state_and_default_time():
    h = Hysteresis(**RULE)
    first = h.update([3, 3])
    second = h.update([3, 3])
    assert first.t.tolist() == [0, 1]
    assert second.t.tolist() == [2, 3]
    assert second.fired.tolist() == [False, True]


def test_update_validates_shapes():
    h = Hysteresis(**RULE)
    with pytest.raises(ValueError, match="одномерн"):
        h.update(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="длины"):
        h.update([1, 2, 3], [0, 1])


def test_empty_series():
    tr = hysteresis([], **RULE)
    assert len(tr) == 0
    assert tr.fire_times.size == 0


def test_trace_fields_and_fire_times():
    tr = hysteresis([0, 3, 3, 3, 3], [0, 1, 2, 3, 4], **RULE)
    assert tr.fired.dtype == np.bool_
    assert tr.fire_times.tolist() == [3.0]
    assert tr.armed.tolist() == [True, True, True, True, False]
    assert tr.since_fired.tolist() == [math.inf] * 4 + [1.0]


def test_reset_and_feed():
    h = Hysteresis(**RULE, armed=False)
    h.update([0, 3, 3, 3])
    h.reset()
    assert h.state == HysteresisState(armed=False)
    steps = list(h.feed([(0, 0), (1, 3), (2, 3), (3, 3)]))
    assert [s.fired for s in steps] == [False, False, False, True]


def test_infinite_index_is_a_value_not_missing():
    assert fires([math.inf] * 4) == [False, False, False, True]
    assert sum(fires([3] * 4 + [-math.inf] + [3] * 10)) == 2


def test_repr_mentions_params():
    assert "upper=2.0" in repr(Hysteresis(**RULE))


def test_rule_locks_when_index_stays_between_thresholds():
    # Перераспределение не устранило дисбаланс полностью: индекс застрял между θ⁻ и θ⁺, затем снова вырос.
    series = [3] * 4 + [1.5] * 30 + [3] * 30
    assert sum(fires(series)) == 1


def test_rearm_after_unlocks_rule():
    series = [3] * 4 + [1.5] * 30 + [3] * 30
    out = fires(series, rearm_after=40)
    # τ = 3, взведение при t ≥ 43. Выдержка набрана ещё при t = 36 (отсчёт от t = 33),
    # поэтому срабатывание происходит в момент взведения.
    assert [k for k, f in enumerate(out) if f] == [3, 43]


def test_rearm_after_respects_cooldown_and_dwell():
    out = fires([3] * 40, rearm_after=0, cooldown=10)
    assert [k for k, f in enumerate(out) if f] == [3, 13, 23, 33]
    out = fires([3] * 40, rearm_after=15, cooldown=10)
    assert [k for k, f in enumerate(out) if f] == [3, 18, 33]


def test_rearm_after_validation():
    with pytest.raises(ValueError, match="rearm_after"):
        hysteresis([1], **RULE, rearm_after=-1)
    with pytest.raises(TypeError, match="rearm_after"):
        hysteresis([1], **RULE, rearm_after="10")
