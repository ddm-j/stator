import math

import numpy as np
import pytest

from strike_fit import conventions as cv


def test_wrap_edges():
    assert cv.wrap(math.pi) == pytest.approx(math.pi)
    assert cv.wrap(-math.pi) == pytest.approx(math.pi)
    assert cv.wrap(3 * math.pi) == pytest.approx(math.pi)
    assert cv.wrap(0.0) == 0.0
    x = np.linspace(-20, 20, 1001)
    w = cv.wrap(x)
    assert np.all(w > -math.pi) and np.all(w <= math.pi)
    assert np.allclose(np.sin(w), np.sin(x)) and np.allclose(np.cos(w), np.cos(x))


def test_wrap_positive():
    x = np.linspace(-20, 20, 501)
    w = cv.wrap_positive(x)
    assert np.all(w >= 0) and np.all(w < cv.TWO_PI)
    assert np.allclose(np.cos(w), np.cos(x))


def test_direction_sign():
    assert cv.direction_sign("cw") == 1 and cv.direction_sign("ccw") == -1
    assert cv.direction_sign(1) == 1 and cv.direction_sign(-1.0) == -1
    with pytest.raises(ValueError):
        cv.direction_sign(0)
    with pytest.raises(KeyError):
        cv.direction_sign("clockwise")


def test_spacing_and_kappa():
    assert cv.deflector_spacing(8) == pytest.approx(math.pi / 4)
    # kappa is stator's A = 1.5 + 2 a^2
    for a in (0.0, 0.0225, 0.05):
        assert cv.kappa(a) == pytest.approx(1.5 + 2 * a * a)


def test_mad_scale_gaussian():
    rng = np.random.default_rng(0)
    assert cv.mad_scale(rng.normal(0.0, 0.05, 20000)) == pytest.approx(0.05, rel=0.05)
