"""Свойства правила на случайных рядах (hypothesis, детерминированный прогон)."""

import math
from fractions import Fraction
from itertools import pairwise

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from reference import reference

from trosna_gisteresis import Hysteresis, HysteresisParams, hysteresis

UPPER, LOWER = 2.0, 1.0
# Значения индекса специально попадают на сами пороги, чтобы проверять строгость неравенств.
values = st.one_of(
    st.sampled_from([0.0, 0.5, LOWER, 1.5, UPPER, 2.5, 3.0]),
    st.floats(-1, 4, allow_nan=False),
    st.just(math.nan),
)
# Интервал мониторинга 1–10 с, как в управляющей системе.
steps = st.integers(1, 10)


@st.composite
def series(draw, min_size=0, max_size=80):
    n = draw(st.integers(min_size, max_size))
    index = draw(st.lists(values, min_size=n, max_size=n))
    start = draw(st.integers(-100, 100))
    t = np.cumsum([start, *draw(st.lists(steps, min_size=max(n - 1, 0), max_size=max(n - 1, 0)))])[:n]
    return index, t.tolist()


params = st.builds(
    HysteresisParams,
    upper=st.just(UPPER),
    lower=st.just(LOWER),
    dwell=st.integers(0, 40),
    cooldown=st.integers(0, 60),
    dwell_from=st.sampled_from(["interval", "sample"]),
    missing=st.sampled_from(["reset", "hold"]),
    armed=st.booleans(),
    rearm_after=st.one_of(st.none(), st.integers(0, 80)),
)


def run(p, index, t):
    return Hysteresis(p).update(index, t)


def kwargs(p):
    return {f: getattr(p, f) for f in HysteresisParams.__dataclass_fields__}


@given(series(), params)
def test_matches_literal_formulas(data, p):
    index, t = data
    fired, dwell, armed = reference(index, t, **kwargs(p))
    tr = run(p, index, t)
    assert tr.fired.tolist() == fired
    assert tr.dwell.tolist() == dwell
    assert tr.armed.tolist() == armed


@given(series(), params)
def test_matches_formulas_with_fractional_times(data, p):
    # Шаг 0.1 с не представим в двоичной арифметике точно; эталон считает в Fraction.
    index, t = data
    exact = [Fraction(k, 10) for k in t]
    pf = HysteresisParams(**{**kwargs(p), "dwell": p.dwell / 10, "cooldown": p.cooldown / 10})
    ref = reference(
        index,
        exact,
        **{**kwargs(p), "dwell": Fraction(int(p.dwell), 10), "cooldown": Fraction(int(p.cooldown), 10)},
    )
    assert run(pf, index, [k / 10 for k in t]).fired.tolist() == ref[0]


@given(series(), params)
def test_fires_are_separated_by_cooldown(data, p):
    tr = run(p, *data)
    assert all(b - a >= p.cooldown for a, b in pairwise(tr.fire_times))


@given(series(), params)
def test_rearm_between_fires(data, p):
    index, t = data
    prev = None
    for k in np.flatnonzero(run(p, index, t).fired):
        if prev is not None or (not p.armed and p.rearm_after is None):
            since = index[(0 if prev is None else prev + 1) : k]
            timed_out = prev is not None and p.rearm_after is not None and t[k] - t[prev] >= p.rearm_after
            assert timed_out or any(v < LOWER for v in since), "срабатывание без взведения"
        prev = k


@given(series(), params)
def test_fire_needs_continuous_exceedance_for_dwell(data, p):
    index, t = data
    tr = run(p, index, t)
    for k in np.flatnonzero(tr.fired):
        assert index[k] > UPPER
        window = [
            j
            for j in range(k + 1)
            if (t[j] >= t[k] - p.dwell if p.dwell_from == "sample" else t[j] > t[k] - p.dwell)
        ]
        for j in window:
            ok = index[j] > UPPER or (p.missing == "hold" and math.isnan(index[j]))
            assert ok, f"точка {j} внутри выдержки не превышает θ⁺"


@given(series(), params)
def test_no_fire_without_exceedance(data, p):
    index, t = data
    clipped = [min(v, UPPER) for v in index]
    assert not run(p, clipped, t).fired.any()


@given(series(), params)
def test_at_most_one_fire_without_drop(data, p):
    index, t = data
    raised = [v if math.isnan(v) else max(v, LOWER) for v in index]
    tr = run(p, raised, t)
    if p.rearm_after is None:
        assert tr.fired.sum() <= (1 if p.armed else 0)
    else:
        # Взведение только по времени: срабатывания разнесены не меньше чем на max(T_c, T_r).
        assert all(b - a >= max(p.cooldown, p.rearm_after) for a, b in pairwise(tr.fire_times))


@given(series(), params, st.integers(1, 79))
def test_chunked_and_restored_equal_whole(data, p, cut):
    index, t = data
    whole = run(p, index, t)
    h = Hysteresis(p)
    head = h.update(index[:cut], t[:cut])
    restored = Hysteresis(p, state=h.state)
    tail = restored.update(index[cut:], t[cut:])
    assert np.concatenate([head.fired, tail.fired]).tolist() == whole.fired.tolist()
    assert np.concatenate([head.dwell, tail.dwell]).tolist() == whole.dwell.tolist()


@given(series(), params)
def test_stream_equals_batch(data, p):
    index, t = data
    h = Hysteresis(p)
    stream = [h.step(v, tk).fired for v, tk in zip(index, t, strict=True)]
    assert stream == hysteresis(index, t, **kwargs(p)).fired.tolist()


@given(series(), params, st.integers(-(10**6), 10**6))
def test_time_shift_invariance(data, p, shift):
    index, t = data
    assert run(p, index, [x + shift for x in t]).fired.tolist() == run(p, index, t).fired.tolist()


@given(series(), params, st.sampled_from([0.25, 0.5, 2.0, 8.0]))
def test_time_scale_invariance(data, p, c):
    index, t = data
    rearm = None if p.rearm_after is None else p.rearm_after * c
    scaled = HysteresisParams(
        **{**kwargs(p), "dwell": p.dwell * c, "cooldown": p.cooldown * c, "rearm_after": rearm}
    )
    assert run(scaled, index, [x * c for x in t]).fired.tolist() == run(p, index, t).fired.tolist()


@given(series(), params, st.sampled_from([0.5, 2.0, 4.0]), st.sampled_from([-3.0, 0.0, 10.0]))
def test_index_affine_invariance(data, p, a, b):
    # Решение зависит только от положения индекса относительно порогов.
    index, t = data
    moved = HysteresisParams(**{**kwargs(p), "upper": a * UPPER + b, "lower": a * LOWER + b})
    assert run(moved, [a * v + b for v in index], t).fired.tolist() == run(p, index, t).fired.tolist()


@given(series(min_size=1), params)
def test_sample_mode_never_fires_earlier_than_interval_mode_first_time(data, p):
    index, t = data
    first = {}
    for mode in ("interval", "sample"):
        tr = run(HysteresisParams(**{**kwargs(p), "dwell_from": mode}), index, t)
        first[mode] = tr.fire_times[0] if tr.fired.any() else math.inf
    assert first["sample"] >= first["interval"]


@given(st.integers(1_600_000_000, 1_900_000_000), st.integers(1, 600), st.sampled_from([0.1, 0.25, 1.0, 2.5]))
def test_large_timestamps_hit_dwell_boundary_exactly(t0, k, dt):
    # Unix-время с шагом 0.1 с: d_t = t − t_start с ошибкой округления не должно пропускать границу T_d.
    t = t0 + dt * np.arange(k + 2)
    tr = hysteresis(np.full(k + 2, 3.0), t, upper=UPPER, lower=LOWER, dwell=k * dt, cooldown=0)
    assert np.flatnonzero(tr.fired).tolist()[:1] == [k]
