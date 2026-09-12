"""Equivalence against stator, for the machinery the two models share.

Anything both models do must be the *same* thing, so that a difference in
predictive power is attributable to the calibration channel and nothing else.
Shared: the per-spin deceleration fit and its pooling, the lap floor, the ``To``
estimator, the exit equations, the rotor fit, the pocket table and the
ball-plus-rotor arithmetic. Not shared, and deliberately so: what the model is
calibrated on, and where it drops the ball.
"""

import math

import numpy as np
import pytest
import stator

import strike_fit
from strike_fit import StrikeRim, synth
from strike_fit.conventions import TWO_PI
from strike_fit.predictor import Predictor

A, B = 0.0225, math.sqrt(10.0 / 3.0)
DELTA, ETA, OMEGA_SQ = 1.0, 0.2, 7.62
M, Y = 4, 6


# ----------------------------------------------------------------------------
# Fixtures: one synthetic corpus, and a rotor
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spins():
    ds = synth.generate(120, eta=0.2, seed=21)
    out = []
    for rec in ds.records:
        s = 1.0 if rec["direction"] == "cw" else -1.0
        st = rec["strike"]
        # what stator would be handed for the same spin: the lab-frame departure
        # angle inside the final revolution
        bin_ = rec["legacy"]["angle_bin"]
        theta_lab = TWO_PI * (bin_ - 0.5) / 36.0
        out.append({"id": rec["spin_id"], "ts": rec["clicks_s"], "s": s,
                    "theta": theta_lab,
                    "t_strike": st["t_s"] if st else None,
                    "deflector": st["deflector"] if st else None})
    return out


def rotor_crossings(v0, k, n_laps, t0=0.0):
    """Times at which a decelerating rotor passes its reference, in radians."""
    j = np.arange(n_laps + 1, dtype=float)
    disc = v0 * v0 - 2.0 * k * TWO_PI * j
    j = j[disc > 0]
    disc = disc[disc > 0]
    return (t0 + (v0 - np.sqrt(disc)) / k).tolist()


@pytest.fixture(scope="module")
def wheel():
    w = stator.Wheel.European()
    for i, (v0, k) in enumerate([(3.3, 0.055), (3.0, 0.050), (3.6, 0.060), (2.8, 0.048)]):
        w.add_timing(f"W{i}", rotor_crossings(v0, k, 8, t0=100.0 * i))
    w.fit()
    assert w.decay_k > 0
    return w


# ----------------------------------------------------------------------------
# The rotor stage is stator's, not a copy of it
# ----------------------------------------------------------------------------

def test_wheel_is_stator_wheel():
    assert strike_fit.Wheel is stator.Wheel


def test_pocket_table_is_the_european_ring(wheel):
    assert wheel.n == 37
    assert wheel.get_pkt_from_angle(0.0) == "0"
    assert wheel.get_pkt_index("0") == 0
    # every pocket resolves back to itself through its own angle
    for name in ("0", "32", "26", "17", "5"):
        assert wheel.get_pkt_from_angle(wheel.get_pkt_angle(name)) == name


# ----------------------------------------------------------------------------
# Backbone: identical inputs must give identical a, b, lap floor and slope
# ----------------------------------------------------------------------------

def test_backbone_matches_stator_rim(spins):
    ref = stator.Rim(M, Y)
    sib = StrikeRim(M=M, Y=Y, n_deflectors=8)
    for sp in spins:
        ref.add_timing(sp["id"], sp["ts"], sp["theta"], sp["s"])
        sib.add_timing(sp["id"], sp["ts"], sp["t_strike"], sp["deflector"], sp["s"])
    ref.fit()
    sib.fit()

    assert sib.params.ball_params.a == ref.params.ball_params.a
    assert sib.params.ball_params.b == ref.params.ball_params.b
    assert sib.lap_floor == ref.lap_floor
    assert sib.a_slope == ref.a_slope
    # and they are real numbers, not both zero by accident
    assert 0.005 < sib.params.ball_params.a < 0.05
    assert 1.0 < sib.params.ball_params.b < 3.0


def test_backbone_ignores_the_calibration_channel(spins):
    """Dropping every strike leaves a, b and the lap floor untouched.

    The deceleration stage reads lap crossings only, so a spin with no strike
    still contributes to it. That is what lets the model use all 208 rc_208
    spins for the backbone while calibrating on the annotated subset.
    """
    with_strikes = StrikeRim(M=M, Y=Y)
    laps_only = StrikeRim(M=M, Y=Y)
    for sp in spins:
        with_strikes.add_timing(sp["id"], sp["ts"], sp["t_strike"], sp["deflector"], sp["s"])
        laps_only.add_timing(sp["id"], sp["ts"], None, None, sp["s"])
    with_strikes.fit()
    with pytest.raises(RuntimeError, match="no spin carries a strike"):
        laps_only.fit()
    # the backbone still ran before the calibration refused
    assert laps_only.lap_floor == with_strikes.lap_floor
    assert laps_only.a_slope == with_strikes.a_slope


# ----------------------------------------------------------------------------
# Exit equations: with no descent, the sibling is stator
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("s", [1.0, -1.0])
def test_rim_exit_matches_stator(spins, s):
    ref = stator.Rim(A, B, DELTA, ETA, OMEGA_SQ)
    sib = StrikeRim.from_params(A, B, DELTA, ETA, OMEGA_SQ, A0_cw=0.0, A0_ccw=0.0, C0=0.0,
                                lap_floor=0.0, a_slope=0.0, M=M, Y=Y)
    compared = 0
    for sp in spins[:40]:
        ts = sp["ts"][: len(sp["ts"]) - M]
        got_ref = ref.predict(ts, s)
        got_sib = sib.predict(ts, s)
        if got_ref is None:
            assert got_sib is None
            continue
        compared += 1
        assert got_sib.theta == pytest.approx(got_ref.theta, abs=1e-12)
        assert got_sib.t_f == pytest.approx(got_ref.t_f, abs=1e-12)
        assert got_sib.theta_rim == got_sib.theta          # no descent arc
        assert got_sib.t_rim == got_sib.t_f
    assert compared > 20


def test_descent_constants_shift_the_prediction_and_nothing_else(spins):
    plain = StrikeRim.from_params(A, B, DELTA, ETA, OMEGA_SQ, 0.0, 0.0, 0.0,
                                  lap_floor=0.0, a_slope=0.0, M=M, Y=Y)
    shifted = StrikeRim.from_params(A, B, DELTA, ETA, OMEGA_SQ, 0.4, 0.9, 0.5,
                                    lap_floor=0.0, a_slope=0.0, M=M, Y=Y)
    for sp in spins[:30]:
        ts = sp["ts"][: len(sp["ts"]) - M]
        for s, arc in ((1.0, 0.4), (-1.0, 0.9)):
            p0, p1 = plain.predict(ts, s), shifted.predict(ts, s)
            if p0 is None:
                continue
            assert p1.theta_rim == pytest.approx(p0.theta_rim, abs=1e-12)
            assert p1.t_rim == pytest.approx(p0.t_rim, abs=1e-12)
            assert p1.theta == pytest.approx(p0.theta + arc, abs=1e-12)
            assert p1.t_f == pytest.approx(p0.t_f + 0.5, abs=1e-12)


# ----------------------------------------------------------------------------
# Predictor: same rotor arithmetic, same pocket
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("ball_sense,wheel_sense", [(1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)])
def test_predictor_matches_stator_with_no_descent(spins, wheel, ball_sense, wheel_sense):
    """Give the sibling stator's drop point and it must give stator's pocket."""
    ref = stator.Predictor(stator.Rim(A, B, DELTA, ETA, OMEGA_SQ), wheel)
    sib = Predictor(StrikeRim.from_params(A, B, DELTA, ETA, OMEGA_SQ, 0.0, 0.0, 0.0,
                                          lap_floor=0.0, a_slope=0.0, M=M, Y=Y), wheel)
    wheel_ts = [0.0, 2.05]
    compared = 0
    for sp in spins[:40]:
        ts = np.asarray(sp["ts"][: len(sp["ts"]) - M]) + 3.0     # off the rotor's zero
        got_ref = ref.predict(ts.tolist(), wheel_ts, ball_sense, wheel_sense)
        got_sib = sib.predict(ts.tolist(), wheel_ts, ball_sense, wheel_sense)
        if got_ref is None:
            assert got_sib is None
            continue
        compared += 1
        assert got_sib.departure_time == pytest.approx(got_ref.departure_time, abs=1e-12)
        assert got_sib.ball_travel == pytest.approx(got_ref.ball_travel, abs=1e-12)
        assert got_sib.wheel_travel == pytest.approx(got_ref.wheel_travel, abs=1e-12)
        assert got_sib.wheel_angle == pytest.approx(got_ref.wheel_angle, abs=1e-12)
        assert got_sib.pocket == got_ref.pocket
    assert compared > 20


def test_descent_moves_the_pocket(spins, wheel):
    """The rotor turns during the descent, so the two models land differently.

    This is the modelling difference, not a bug: stator advances the rotor to
    rim exit, this model advances it to the strike.
    """
    ref = stator.Predictor(stator.Rim(A, B, DELTA, ETA, OMEGA_SQ), wheel)
    sib = Predictor(StrikeRim.from_params(A, B, DELTA, ETA, OMEGA_SQ, 0.5, 0.5, 0.45,
                                          lap_floor=0.0, a_slope=0.0, M=M, Y=Y), wheel)
    wheel_ts = [0.0, 2.05]
    moved = 0
    total = 0
    for sp in spins[:40]:
        ts = (np.asarray(sp["ts"][: len(sp["ts"]) - M]) + 3.0).tolist()
        got_ref = ref.predict(ts, wheel_ts, 1.0, -1.0)
        got_sib = sib.predict(ts, wheel_ts, 1.0, -1.0)
        if got_ref is None:
            continue
        total += 1
        assert got_sib.departure_time == pytest.approx(got_ref.departure_time + 0.45, abs=1e-9)
        assert got_sib.wheel_travel > got_ref.wheel_travel
        moved += got_sib.pocket != got_ref.pocket
    assert total > 20
    assert moved > 0.8 * total
