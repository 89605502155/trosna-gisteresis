"""Сравнение постоянного порога и гистерезиса с выдержкой и охлаждением на модели трёх ВМ.

Свободная память трёх ВМ меняется под нагрузкой: короткоживущие процессы дают кратковременные
выбросы, длительная нагрузка — устойчивый дисбаланс. Индекс неравномерности — коэффициент
вариации долей свободной памяти. После решения память выравнивается (перераспределение).

Запуск: ``uv run python examples/demo.py`` — график сохраняется в ``examples/figures``.
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trosna_gisteresis import Hysteresis

OUT = Path(__file__).parent / "figures"
SEED = 20261011
TOTAL = np.array([4096.0, 8192.0, 2048.0])  # МБ
UPPER, LOWER, DWELL, COOLDOWN = 0.15, 0.08, 30.0, 120.0


def load(t: float, rng: np.random.Generator) -> np.ndarray:
    """Занятая память ВМ, МБ."""
    base = np.array([0.4, 0.4, 0.4]) * TOTAL
    if 600 <= t < 1500:  # длительная нагрузка на третью ВМ
        base[2] += 0.45 * TOTAL[2]
    if rng.random() < 0.03:  # короткоживущий процесс
        base[rng.integers(3)] += 0.5 * TOTAL[2]
    return base + 30 * rng.standard_normal(3)


def simulate(decide) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = np.random.default_rng(SEED)
    t, times, index, fires = 0.0, [], [], []
    shift = np.zeros(3)  # перераспределённая память, МБ
    while t < 2400:
        used = load(t, rng)
        alloc = TOTAL + shift
        phi = np.clip(alloc - used, 0, None) / alloc
        cv = float(np.std(phi) / np.mean(phi))
        times.append(t)
        index.append(cv)
        if decide(cv, t):
            fires.append(t)
            free_target = (alloc.sum() - used.sum()) / alloc.sum()
            shift = used / (1 - free_target) - TOTAL  # выравнивание долей свободной памяти
        t += float(rng.integers(1, 11))  # интервал мониторинга 1–10 с
    return np.array(times), np.array(index), fires


def main() -> None:
    OUT.mkdir(exist_ok=True)
    t1, i1, single = simulate(lambda cv, _t: cv > UPPER)
    rule = Hysteresis(upper=UPPER, lower=LOWER, dwell=DWELL, cooldown=COOLDOWN)
    t2, i2, hyst = simulate(lambda cv, t: rule.step(cv, t).fired)
    print(f"постоянный порог: {len(single)} перераспределений")
    print(f"гистерезис:       {len(hyst)} перераспределений, моменты: {hyst}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, t, idx, fires, title in [
        (axes[0], t1, i1, single, "Постоянный порог"),
        (axes[1], t2, i2, hyst, "Гистерезис с выдержкой и охлаждением"),
    ]:
        ax.plot(t, idx, lw=0.8, color="0.3", label="индекс (CV)")
        ax.axhline(UPPER, color="C3", ls="--", lw=1, label="θ⁺")
        if ax is axes[1]:
            ax.axhline(LOWER, color="C0", ls="--", lw=1, label="θ⁻")
        for f in fires:
            ax.axvline(f, color="C2", alpha=0.4, lw=1)
        ax.set_title(f"{title}: {len(fires)} перераспределений")
        ax.set_ylabel("индекс")
        ax.legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("время, с")
    fig.tight_layout()
    fig.savefig(OUT / "threshold_vs_hysteresis.png", dpi=150)


if __name__ == "__main__":
    main()
