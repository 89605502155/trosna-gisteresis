"""Эталон: буквальная запись формул (9)–(11) с накоплением d_t += Δt_t.

Используется только в тестах. Время передаётся целыми числами или ``Fraction``, поэтому
арифметика точная и сравнение с библиотекой не зависит от округления.
"""

import math


def reference(
    index,
    t,
    *,
    upper,
    lower,
    dwell,
    cooldown,
    dwell_from="interval",
    missing="reset",
    armed=True,
    rearm_after=None,
):
    d = 0
    run = False  # предыдущая непропущенная точка была выше θ⁺ (серия продолжается)
    g = bool(armed)
    tau = None
    prev = None
    fired, dwells, armeds = [], [], []
    for value, tk in zip(index, t, strict=True):
        dt = 0 if prev is None else tk - prev
        prev = tk
        if math.isnan(value):
            if missing == "reset":
                d, run = 0, False
            elif run:
                d += dt
            fired.append(False)
            dwells.append(d)
            armeds.append(g)
            continue
        above = value > upper
        if above:
            if run:
                d += dt
            else:
                d = dt if dwell_from == "interval" else 0  # (9) при d_{t-1} = 0
            run = True
        else:
            d, run = 0, False
        rearm = rearm_after is not None and (tau is None or tk - tau >= rearm_after)
        g = g or value < lower or rearm  # (10), (10′)
        f = above and d >= dwell and g and (tau is None or tk - tau >= cooldown)  # (11)
        fired.append(f)
        dwells.append(d)
        armeds.append(g)
        if f:
            tau, g, d, run = tk, False, 0, False
    return fired, dwells, armeds
