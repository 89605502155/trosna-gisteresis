"""Гистерезис с выдержкой времени и периодом охлаждения.

Правило решения о перераспределении по индексу неравномерности ``I_t``:

    d_t = (d_{t-1} + Δt_t) · [I_t > θ⁺]                        (9)
    g_t = max{g_{t-1}, [I_t < θ⁻]}                               (10)
    δ_t = 1  ⇔  I_t > θ⁺ ∧ d_t ≥ T_d ∧ g_t = 1 ∧ t − τ ≥ T_c     (11)

После срабатывания ``τ = t``, ``g_t = 0``, ``d_t = 0``.

Необязательное расширение (параметр ``rearm_after = T_r``): флаг готовности взводится
также по истечении времени ``T_r`` после срабатывания,

    g_t = max{g_{t-1}, [I_t < θ⁻], [t − τ ≥ T_r]},                (10′)

чтобы правило не блокировалось, если индекс после неудачного перераспределения
не опускается ниже ``θ⁻``.

Время непрерывного превышения хранится не накоплением ``d += Δt``, а как момент начала серии
превышений: ``d_t = t − t_start``. Это то же самое, но без накопления ошибки округления
на тысячах шагов и при больших отметках времени (Unix-время).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

DwellFrom = Literal["interval", "sample"]
Missing = Literal["reset", "hold"]

_DWELL_FROM = ("interval", "sample")
_MISSING = ("reset", "hold")
# Допуск сравнения времён в единицах ulp: d_t = t − t_start и t − τ вычисляются с ошибкой
# в несколько ulp(t), её не должно хватать, чтобы пропустить срабатывание ровно на границе T_d или T_c.
_ULP_TOLERANCE = 16


def _reached(elapsed: float, limit: float, t: float) -> bool:
    """``elapsed ≥ limit`` с допуском на ошибку округления отметок времени."""
    return elapsed >= limit - _ULP_TOLERANCE * math.ulp(max(abs(t), 1.0))


@dataclass(frozen=True, slots=True)
class HysteresisParams:
    """Параметры правила (9)–(11).

    Attributes
    ----------
    upper:
        Верхний порог ``θ⁺``: дисбаланс считается значимым при ``I_t > θ⁺``.
    lower:
        Нижний порог ``θ⁻ < θ⁺``: после срабатывания правило снова взводится только при ``I_t < θ⁻``.
    dwell:
        Выдержка ``T_d`` — сколько времени (в единицах ``t``, обычно секунды) индекс должен
        непрерывно превышать ``θ⁺``.
    cooldown:
        Период охлаждения ``T_c`` — минимальное время между двумя срабатываниями.
    dwell_from:
        С какого момента отсчитывается выдержка.
        ``"interval"`` — как в (9): первая точка выше ``θ⁺`` сразу добавляет весь интервал
        ``Δt_t`` от предыдущей точки (верхняя оценка длительности превышения).
        ``"sample"`` — отсчёт от первой точки выше ``θ⁺`` (нижняя оценка).
        Истинная длительность лежит между ними, разница не больше одного интервала мониторинга.
    missing:
        Что делать с пропуском (``NaN``) индекса. ``"reset"`` — непрерывность превышения не
        подтверждена, выдержка обнуляется. ``"hold"`` — считается, что состояние не изменилось,
        выдержка продолжает идти. На пропуске правило никогда не срабатывает и не взводится.
    armed:
        Начальное значение флага готовности ``g_0``. ``True`` — первое срабатывание возможно
        без предварительного снижения индекса ниже ``θ⁻``.
    rearm_after:
        Время ``T_r`` после срабатывания, по истечении которого правило взводится и без
        снижения индекса ниже ``θ⁻`` (формула 10′). ``None`` (по умолчанию) — только по ``θ⁻``,
        как в исходной формулировке. Пока срабатываний не было, ``t − τ = ∞`` и условие выполнено.
    """

    upper: float
    lower: float
    dwell: float
    cooldown: float
    dwell_from: DwellFrom = "interval"
    missing: Missing = "reset"
    armed: bool = True
    rearm_after: float | None = None

    def __post_init__(self) -> None:
        for name in ("upper", "lower", "dwell", "cooldown"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
                raise TypeError(f"{name} должен быть числом, получено {type(value).__name__}")
            if not math.isfinite(value):
                raise ValueError(f"{name} должен быть конечным, получено {value}")
            object.__setattr__(self, name, float(value))
        if not self.upper > self.lower:
            raise ValueError(f"нужно upper > lower (θ⁺ > θ⁻), получено {self.upper} и {self.lower}")
        if self.dwell < 0:
            raise ValueError(f"dwell (T_d) не может быть отрицательной, получено {self.dwell}")
        if self.cooldown < 0:
            raise ValueError(f"cooldown (T_c) не может быть отрицательным, получено {self.cooldown}")
        if self.dwell_from not in _DWELL_FROM:
            raise ValueError(f"dwell_from должен быть одним из {_DWELL_FROM}, получено {self.dwell_from!r}")
        if self.missing not in _MISSING:
            raise ValueError(f"missing должен быть одним из {_MISSING}, получено {self.missing!r}")
        object.__setattr__(self, "armed", bool(self.armed))
        if self.rearm_after is not None:
            value = self.rearm_after
            if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
                raise TypeError(f"rearm_after должен быть числом или None, получено {type(value).__name__}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"rearm_after (T_r) должен быть конечным и неотрицательным, получено {value}"
                )
            object.__setattr__(self, "rearm_after", float(value))


@dataclass(frozen=True, slots=True)
class HysteresisState:
    """Внутреннее состояние правила между точками мониторинга.

    Его можно сохранить и восстановить (например, при перезапуске управляющей системы):
    ``Hysteresis(params, state=old.state)`` продолжает работу с того же места.

    Attributes
    ----------
    t:
        Время последней обработанной точки (``None`` — точек ещё не было).
    run_start:
        Момент начала текущей серии превышений ``θ⁺``, так что ``d_t = t − run_start``
        (``None`` — превышения нет, ``d_t = 0``).
    armed:
        Флаг готовности ``g_t``.
    last_fired:
        Момент последнего срабатывания ``τ`` (``-inf`` — срабатываний не было).
    """

    t: float | None = None
    run_start: float | None = None
    armed: bool = True
    last_fired: float = -math.inf

    @property
    def dwell(self) -> float:
        """Текущее время непрерывного превышения ``d_t``."""
        if self.t is None or self.run_start is None:
            return 0.0
        return self.t - self.run_start


@dataclass(frozen=True, slots=True)
class Step:
    """Результат обработки одной точки.

    ``dwell``, ``armed`` и ``since_fired`` — значения, по которым проверялось условие (11),
    то есть до сброса, выполняемого после срабатывания.
    """

    t: float
    index: float
    fired: bool
    dwell: float
    armed: bool
    since_fired: float


@dataclass(frozen=True)
class Trace:
    """Результат обработки ряда: массивы той же длины, что и вход."""

    t: NDArray[np.float64]
    index: NDArray[np.float64]
    fired: NDArray[np.bool_]
    dwell: NDArray[np.float64]
    armed: NDArray[np.bool_]
    since_fired: NDArray[np.float64]
    state: HysteresisState = field(repr=False)

    @property
    def fire_times(self) -> NDArray[np.float64]:
        """Моменты срабатываний."""
        return self.t[self.fired]

    def __len__(self) -> int:
        return len(self.t)


class Hysteresis:
    """Потоковое правило (9)–(11): по одной точке мониторинга за вызов.

    >>> h = Hysteresis(upper=0.3, lower=0.1, dwell=10.0, cooldown=60.0)
    >>> [h.step(i, t).fired for t, i in [(0, 0.5), (5, 0.5), (10, 0.5), (15, 0.5)]]
    [False, False, True, False]
    """

    __slots__ = ("_state", "params")

    def __init__(
        self,
        params: HysteresisParams | None = None,
        *,
        state: HysteresisState | None = None,
        **kwargs: object,
    ) -> None:
        if params is None:
            params = HysteresisParams(**kwargs)  # type: ignore[arg-type]
        elif kwargs:
            raise TypeError("передайте либо params, либо параметры по отдельности, но не всё вместе")
        self.params: HysteresisParams = params
        self._state = state if state is not None else HysteresisState(armed=params.armed)

    @property
    def state(self) -> HysteresisState:
        return self._state

    def reset(self) -> None:
        """Вернуть правило в начальное состояние."""
        self._state = HysteresisState(armed=self.params.armed)

    def step(self, index: float, t: float) -> Step:
        """Обработать значение индекса ``index`` в момент ``t``; вернуть решение ``δ_t``."""
        p, s = self.params, self._state
        t = float(t)
        index = float(index)
        if not math.isfinite(t):
            raise ValueError(f"время должно быть конечным, получено {t}")
        if s.t is not None and not t > s.t:
            raise ValueError(f"время должно строго возрастать: {t} после {s.t}")

        run_start, armed = s.run_start, s.armed
        if math.isnan(index):
            if p.missing == "reset":
                run_start = None
            dwell = 0.0 if run_start is None else t - run_start
            self._state = HysteresisState(t, run_start, armed, s.last_fired)
            return Step(t, index, False, dwell, armed, t - s.last_fired)

        above = index > p.upper
        if not above:
            run_start = None
        elif run_start is None:
            # (9): d_t = d_{t-1} + Δt_t при d_{t-1} = 0. Для самой первой точки Δt не определён, берём 0.
            run_start = s.t if p.dwell_from == "interval" and s.t is not None else t
        since = t - s.last_fired
        if index < p.lower or (p.rearm_after is not None and _reached(since, p.rearm_after, t)):
            armed = True

        dwell = 0.0 if run_start is None else t - run_start
        # I_t > θ⁺ проверяется явно: при T_d = 0 условие d_t ≥ T_d выполняется и без превышения.
        fired = above and armed and _reached(dwell, p.dwell, t) and _reached(since, p.cooldown, t)
        if fired:
            self._state = HysteresisState(t, None, False, t)
        else:
            self._state = HysteresisState(t, run_start, armed, s.last_fired)
        return Step(t, index, fired, dwell, armed, since)

    def update(self, index: ArrayLike, t: ArrayLike | None = None) -> Trace:
        """Обработать ряд точек подряд (продолжая текущее состояние).

        Если ``t`` не задано, точки считаются идущими через 1 единицу времени после последней
        обработанной (с 0, если точек ещё не было).
        """
        idx = np.asarray(index, dtype=np.float64)
        if idx.ndim != 1:
            raise ValueError(f"index должен быть одномерным, получена размерность {idx.ndim}")
        if t is None:
            start = 0.0 if self._state.t is None else self._state.t + 1.0
            ts = start + np.arange(len(idx), dtype=np.float64)
        else:
            ts = np.asarray(t, dtype=np.float64)
            if ts.shape != idx.shape:
                raise ValueError(f"длины index и t различаются: {idx.shape} и {ts.shape}")

        n = len(idx)
        fired = np.zeros(n, dtype=np.bool_)
        dwell = np.zeros(n, dtype=np.float64)
        armed = np.zeros(n, dtype=np.bool_)
        since = np.zeros(n, dtype=np.float64)
        for k in range(n):
            r = self.step(idx[k], ts[k])
            fired[k], dwell[k], armed[k], since[k] = r.fired, r.dwell, r.armed, r.since_fired
        return Trace(ts, idx, fired, dwell, armed, since, self._state)

    def feed(self, points: Iterable[tuple[float, float]]) -> Iterable[Step]:
        """Ленивый вариант: ``points`` — пары ``(t, index)``."""
        for t, index in points:
            yield self.step(index, t)

    def __repr__(self) -> str:
        return f"Hysteresis({self.params!r}, state={self._state!r})"


def hysteresis(
    index: ArrayLike,
    t: ArrayLike | None = None,
    *,
    upper: float,
    lower: float,
    dwell: float,
    cooldown: float,
    dwell_from: DwellFrom = "interval",
    missing: Missing = "reset",
    armed: bool = True,
    rearm_after: float | None = None,
) -> Trace:
    """Применить правило (9)–(11) к ряду индекса ``index`` с отметками времени ``t``.

    Без ``t`` точки считаются равноотстоящими с шагом 1 (тогда ``dwell`` и ``cooldown``
    задаются в шагах мониторинга).
    """
    params = HysteresisParams(upper, lower, dwell, cooldown, dwell_from, missing, armed, rearm_after)
    return Hysteresis(params).update(index, t)
