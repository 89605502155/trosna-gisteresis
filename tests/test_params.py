import math

import numpy as np
import pytest

from trosna_gisteresis import Hysteresis, HysteresisParams

BASE = {"upper": 0.3, "lower": 0.1, "dwell": 10.0, "cooldown": 60.0}


def make(**changes):
    return HysteresisParams(**{**BASE, **changes})


def test_defaults_follow_article():
    p = make()
    assert (p.dwell_from, p.missing, p.armed) == ("interval", "reset", True)


def test_numbers_are_converted_to_float():
    p = make(upper=np.float32(0.5), lower=0, dwell=np.int64(3), cooldown=1)
    assert all(type(v) is float for v in (p.upper, p.lower, p.dwell, p.cooldown))


@pytest.mark.parametrize(("upper", "lower"), [(0.1, 0.1), (0.1, 0.3)])
def test_upper_must_exceed_lower(upper, lower):
    with pytest.raises(ValueError, match="upper > lower"):
        make(upper=upper, lower=lower)


@pytest.mark.parametrize("name", ["upper", "lower", "dwell", "cooldown"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_rejected(name, value):
    with pytest.raises(ValueError, match="конечным"):
        make(**{name: value})


@pytest.mark.parametrize("name", ["upper", "dwell"])
@pytest.mark.parametrize("value", ["1", None, True])
def test_non_numbers_rejected(name, value):
    with pytest.raises(TypeError, match="числом"):
        make(**{name: value})


@pytest.mark.parametrize("name", ["dwell", "cooldown"])
def test_negative_times_rejected(name):
    with pytest.raises(ValueError, match="отрицательн"):
        make(**{name: -1.0})


def test_zero_times_allowed():
    p = make(dwell=0, cooldown=0)
    assert p.dwell == p.cooldown == 0.0


@pytest.mark.parametrize(("name", "value"), [("dwell_from", "start"), ("missing", "skip")])
def test_unknown_options_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        make(**{name: value})


def test_params_are_immutable():
    with pytest.raises(AttributeError):
        make().upper = 1.0  # type: ignore[misc]


def test_constructor_accepts_params_or_kwargs_but_not_both():
    assert Hysteresis(make()).params == Hysteresis(**BASE).params
    with pytest.raises(TypeError, match="либо params"):
        Hysteresis(make(), upper=0.5)
