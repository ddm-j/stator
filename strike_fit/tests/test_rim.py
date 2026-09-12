"""The model's own behaviour: geometry, interface, and recovery of truth."""

import math

import numpy as np
import pytest
import stator

from strike_fit import StrikeRim, synth
from strike_fit.calibrate import CalibrationConfig
from strike_fit.conventions import TWO_PI
from strike_fit.rim import (deflector_from_travel_angle, deflector_travel_angle,
                            strike_deflector)

M, Y = 4, 6
SPACING = TWO_PI / 8


# ----------------------------------------------------------------------------
# Deflector geometry
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("s", [1, -1])
def test_deflector_angles_span_a_full_revolution(s):
    angles = [deflector_travel_angle(d, s, 8) for d in range(8)]
    assert sorted(angles) == pytest.approx([(i + 1) * SPACING for i in range(8)])
    # the reference deflector is a whole revolution away, not none
    assert deflector_travel_angle(0, s, 8) == pytest.approx(TWO_PI)
    assert len(set(np.round(angles, 9))) == 8


def test_senses_are_mirror_images():
    for d in range(1, 8):
        assert (deflector_travel_angle(d, 1, 8)
                + deflector_travel_angle(d, -1, 8)) == pytest.approx(TWO_PI)


def test_deflector_out_of_range():
    with pytest.raises(ValueError):
        deflector_travel_angle(8, 1, 8)
    with pytest.raises(ValueError):
        deflector_travel_angle(-1, 1, 8)


@pytest.mark.parametrize("s", [1, -1])
def test_strike_deflector_round_trip(s):
    """A ball just short of a deflector strikes it, and the angle reconstructs."""
    for n_extra in range(4):
        for d in range(8):
            psi = deflector_travel_angle(d, s, 8)
            theta_s = TWO_PI * n_extra + psi - 0.3 * SPACING
            got_d, got_n = strike_deflector(theta_s, s, 8)
            assert got_d == d
            assert got_n == n_extra
            back = deflector_travel_angle(got_d, s, 8) + TWO_PI * got_n
            assert 0.0 < back - theta_s < SPACING


@pytest.mark.parametrize("s", [1, -1])
def test_nearest_deflector_is_the_prediction_map(s):
    for d in range(8):
        centre = deflector_travel_angle(d, s, 8)
        for off in (-0.4 * SPACING, 0.0, 0.4 * SPACING):
            assert deflector_from_travel_angle(centre + off, s, 8) == d
        assert deflector_from_travel_angle(centre + TWO_PI, s, 8) == d


# ----------------------------------------------------------------------------
# Interface
# ----------------------------------------------------------------------------

def clean_spin(omega0=12.0, n=13, a=0.0225, b=math.sqrt(10 / 3), t0=17.0):
    from strike_fit import model
    return (model.lap_times(np.arange(n), a, b, omega0) + t0).tolist()


def test_add_timing_validation():
    rim = StrikeRim(M=M, Y=Y)
    ts = clean_spin()
    with pytest.raises(ValueError, match="must be \\+1 or -1"):
        rim.add_timing("a", ts, ts[-1] + 1, 3, 0.0)
    with pytest.raises(ValueError, match="more than two timestamps"):
        rim.add_timing("a", ts[:2], ts[1] + 1, 3, 1.0)
    with pytest.raises(ValueError, match="non-monotonic"):
        rim.add_timing("a", [0.0, 2.0, 1.0, 3.0], 9.0, 3, 1.0)
    with pytest.raises(ValueError, match="outside 0..7"):
        rim.add_timing("a", ts, ts[-1] + 1, 8, 1.0)
    with pytest.raises(ValueError, match="must follow the last crossing"):
        rim.add_timing("a", ts, ts[-1] - 0.1, 3, 1.0)
    with pytest.raises(ValueError, match="both a strike time and a deflector"):
        rim.add_timing("a", ts, ts[-1] + 1, None, 1.0)
    assert rim.n_timings == 0


def test_add_timing_records_travel_and_relative_times():
    rim = StrikeRim(M=M, Y=Y)
    ts = clean_spin(t0=17.0)
    rim.add_timing("a", ts, ts[-1] + 0.8, 3, 1.0)
    rec = rim._data[0]
    assert rec.tk[0] == 0.0
    assert rec.t_strike == pytest.approx(ts[-1] + 0.8 - ts[0])
    assert rec.theta == pytest.approx(deflector_travel_angle(3, 1, 8) + TWO_PI * (len(ts) - 1))
    assert rec.has_strike


def test_add_timing_batch_matches_single_calls():
    ts = [clean_spin(omega0=o) for o in (11.0, 12.0, 13.0)]
    strikes = [t[-1] + 0.7 for t in ts]
    defl = [1, 4, 6]
    senses = [1.0, -1.0, 1.0]
    one = StrikeRim(M=M, Y=Y)
    for j in range(3):
        one.add_timing(f"s{j}", ts[j], strikes[j], defl[j], senses[j])
    many = StrikeRim(M=M, Y=Y)
    many.add_timing([f"s{j}" for j in range(3)], ts, strikes, defl, senses)
    assert [r.theta for r in many._data] == [r.theta for r in one._data]
    assert [r.s for r in many._data] == [r.s for r in one._data]


def test_predict_requires_a_fit_and_enough_crossings():
    rim = StrikeRim(M=M, Y=Y)
    with pytest.raises(RuntimeError, match="has not been fit"):
        rim.predict(clean_spin(), 1.0)
    fitted = StrikeRim.from_params(0.0225, math.sqrt(10 / 3), 1.0, 0.2, 7.62, M=M, Y=Y)
    with pytest.raises(ValueError, match="crossings for a 6 lap window"):
        fitted.predict(clean_spin()[:4], 1.0)
    with pytest.raises(ValueError, match="non-monotonic"):
        fitted.predict([0.0, 2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0], 1.0)


def test_params_shape_mirrors_stator():
    rim = StrikeRim.from_params(0.0225, 1.8257, 1.0, 0.2, 7.62, 0.5, 0.7, 0.45, M=M, Y=Y)
    p = rim.params
    assert p.ball_params.a == 0.0225 and p.ball_params.b == 1.8257
    assert p.dep_params.delta == 1.0 and p.dep_params.eta == 0.2
    assert p.dep_params.omega_sq == 7.62
    assert p.descent_params.arc(1.0) == 0.5 and p.descent_params.arc(-1.0) == 0.7
    # the rim block is a real stator.FitParams and stator functions accept it
    assert isinstance(p.fit_params, stator.FitParams)
    assert stator.predict_theta(1.1, 1.0, p.fit_params) is not None


def test_batch_predict_matches_single_calls():
    rim = StrikeRim.from_params(0.0225, math.sqrt(10 / 3), 1.0, 0.2, 7.62, 0.5, 0.7, 0.45,
                                M=M, Y=Y)
    tss = [clean_spin(omega0=o)[:-M] for o in (11.0, 12.5, 14.0)]
    senses = [1.0, -1.0, 1.0]
    batch = rim.predict(tss, senses)
    assert len(batch) == 3
    for got, ts, s in zip(batch, tss, senses):
        one = rim.predict(ts, s)
        assert got.theta == one.theta and got.t_f == one.t_f
        assert got.deflector == one.deflector


# ----------------------------------------------------------------------------
# Recovery of known truth
# ----------------------------------------------------------------------------

def load(rim, records):
    for r in records:
        st = r["strike"]
        rim.add_timing(r["spin_id"], r["clicks_s"],
                       st["t_s"] if st else None, st["deflector"] if st else None,
                       1.0 if r["direction"] == "cw" else -1.0)
    return rim


def held_out(rim, records):
    """Predict each spin from crossings ending M laps before its strike."""
    hits = 0
    total = 0
    r_t = []
    for r in records:
        st = r["strike"]
        ts = r["clicks_s"]
        if st is None or len(ts) < rim.M + rim.Y + 1:
            continue
        anchor = len(ts) - 1 - rim.M
        pred = rim.predict(ts[:anchor + 1], 1.0 if r["direction"] == "cw" else -1.0)
        if pred is None:
            continue
        total += 1
        hits += pred.deflector == st["deflector"]
        r_t.append(st["t_s"] - (pred.t_f + ts[anchor]))
    return hits / total, total, np.array(r_t)


@pytest.fixture(scope="module")
def fitted():
    ds = synth.generate(300, eta=0.2, seed=5, mode="mechanical")
    train = ds.records[0::2]
    test = ds.records[1::2]
    rim = load(StrikeRim(M=M, Y=Y, n_deflectors=8,
                         config=CalibrationConfig(annotation_mode="mechanical")), train)
    rim.fit()
    return ds, rim, test


def test_recovers_truth(fitted):
    """What is identified is the backbone, the tilt, and the prediction.

    ``delta``, ``omega_sq`` and the descent offsets are not separately
    identified: the objective has a mirror basin near ``delta + pi`` with the
    offsets shifted to match, and the offsets additionally absorb the ``To``
    estimator's bias. Asserting them individually would be asserting which
    basin the search happened to land in, so this pins the parts that are real
    and leaves prediction quality to the held-out test.
    """
    ds, rim, _ = fitted
    tr = synth.truth_from_header(ds.header)
    p = rim.params
    assert p.ball_params.a == pytest.approx(tr.a, rel=0.05)
    assert p.ball_params.b == pytest.approx(tr.b, rel=0.05)
    # real tilt is detected, and not wildly overstated
    assert 0.5 * tr.eta < p.dep_params.eta < 2.0 * tr.eta
    assert p.dep_params.omega_sq > p.ball_params.b ** 2
    assert rim.calibration.n_failed == 0
    assert len(rim.fitted_ids) > 0.9 * len(rim._data)
    # the offsets stay in a range an effective constant can plausibly occupy
    assert abs(p.descent_params.A0_cw) < 2 * TWO_PI
    assert -1.0 <= p.descent_params.C0 <= 3.0


def test_holding_the_threshold_is_supported(fitted):
    """The exit threshold can be held instead of fitted, and still predicts.

    stator fits it, so this fits it by default. Holding it is offered because
    it is the paper's convention, and an experiment may want to compare.
    """
    ds, _, test = fitted
    cfg = CalibrationConfig(annotation_mode="mechanical", fit_omega_sq=False)
    rim = load(StrikeRim(M=M, Y=Y, config=cfg), ds.records[0::2])
    rim.fit()
    assert rim.params.dep_params.omega_sq == cfg.omega_sq_fixed
    hit, n, r_t = held_out(rim, test)
    assert hit > 0.5 and n > 100
    assert np.median(np.abs(r_t)) < 0.12


def test_held_out_prediction(fitted):
    ds, rim, test = fitted
    hit, n, r_t = held_out(rim, test)
    assert n > 100
    assert hit > 0.6                       # oracle sits near 0.89 on this corpus
    assert abs(np.median(r_t)) < 0.05
    assert np.median(np.abs(r_t)) < 0.08
    # the hover injection shows up as a right tail, not as a shifted centre
    assert (r_t > 1.0).mean() < 0.15


def test_null_tilt_produces_no_edge():
    ds = synth.generate(250, eta=0.0, seed=11, mode="mechanical")
    rim = load(StrikeRim(M=M, Y=Y, config=CalibrationConfig(annotation_mode="mechanical")),
               ds.records)
    rim.fit()
    assert rim.params.dep_params.eta < 0.06
    d = [rim.predict(r["clicks_s"][: len(r["clicks_s"]) - M],
                     1.0 if r["direction"] == "cw" else -1.0).deflector
         for r in ds.records if len(r["clicks_s"]) >= M + Y + 1]
    counts = np.bincount(d, minlength=8)
    assert counts.max() / counts.sum() < 0.45      # no fabricated favourite


def test_spins_without_strikes_still_feed_the_backbone():
    ds = synth.generate(120, eta=0.2, seed=13, mode="mechanical")
    full = load(StrikeRim(M=M, Y=Y), ds.records)
    full.fit()
    half = StrikeRim(M=M, Y=Y)
    for i, r in enumerate(ds.records):
        st = r["strike"] if i % 3 == 0 else None
        half.add_timing(r["spin_id"], r["clicks_s"],
                        st["t_s"] if st else None, st["deflector"] if st else None,
                        1.0 if r["direction"] == "cw" else -1.0)
    half.fit()
    assert half.params.ball_params.a == full.params.ball_params.a
    assert half.lap_floor == full.lap_floor
    assert len(half.fitted_ids) < 0.5 * len(full.fitted_ids)


def test_uncalibrated_direction_refuses_rather_than_guessing():
    """A direction with no strikes must not get a fabricated descent arc.

    On a table where the dealer reverses every round, splitting spins by
    alternating index selects a single direction. Silently predicting the
    other one from a default offset produces confident nonsense, so it raises.
    """
    ds = synth.generate(120, eta=0.2, seed=17, mode="mechanical")
    rim = StrikeRim(M=M, Y=Y, config=CalibrationConfig(annotation_mode="mechanical"))
    for r in ds.records:
        st = r["strike"]
        cw = r["direction"] == "cw"
        rim.add_timing(r["spin_id"], r["clicks_s"],
                       st["t_s"] if (st and cw) else None,
                       st["deflector"] if (st and cw) else None,
                       1.0 if cw else -1.0)
    rim.fit()
    assert rim.calibrated_senses == [1]
    assert rim.calibration.n_by_direction[-1] == 0
    assert math.isnan(rim.params.descent_params.A0_ccw)

    ts = [r["clicks_s"] for r in ds.records if len(r["clicks_s"]) >= M + Y + 1][0]
    assert rim.predict(ts[:-M], 1.0) is not None
    with pytest.raises(ValueError, match="no strike was calibrated"):
        rim.predict(ts[:-M], -1.0)
