"""The fold driver refits without the held-out spins and predicts them live.

A fold must be exactly a ``StrikeRim`` fitted on the other spins, and the
worker count must not change a single number.
"""

import math

import numpy as np
import pytest

from strike_fit import StrikeRim, synth
from strike_fit.calibrate import CalibrationConfig
from strike_fit.crossval import cross_validate, leave_one_out, summary

M, Y = 4, 6

# a coarse grid keeps a fold around a tenth of a second
CFG = CalibrationConfig(annotation_mode="mechanical",
                        delta_grid=np.deg2rad(np.arange(0.0, 360.0, 45.0)),
                        eta_grid=np.array([0.0, 0.2, 0.4]),
                        omega_sq_grid=np.array([6.5, 7.5, 8.5]), n_starts=2)


@pytest.fixture(scope="module")
def spins():
    ds = synth.generate(60, eta=0.2, seed=8, mode="mechanical")
    recs = ds.records
    return dict(
        ids=[r["spin_id"] for r in recs],
        timestamps=[r["clicks_s"] for r in recs],
        t_strikes=[r["strike"]["t_s"] if r["strike"] else None for r in recs],
        deflectors=[r["strike"]["deflector"] if r["strike"] else None for r in recs],
        ss=[1.0 if r["direction"] == "cw" else -1.0 for r in recs],
    )


def args(sp):
    return sp["ids"], sp["timestamps"], sp["t_strikes"], sp["deflectors"], sp["ss"]


def test_a_fold_is_a_rim_fitted_without_the_held_out_spin(spins):
    held = [spins["ids"][3], spins["ids"][10]]
    got = cross_validate(*args(spins), [held], M=M, Y=Y, config=CFG, workers=1)

    rim = StrikeRim(M=M, Y=Y, config=CFG)
    for j, id_ in enumerate(spins["ids"]):
        if id_ not in held:
            rim.add_timing(id_, spins["timestamps"][j], spins["t_strikes"][j],
                           spins["deflectors"][j], spins["ss"][j])
    rim.fit()

    assert [r.id for r in got] == held
    for r in got:
        j = spins["ids"].index(r.id)
        ts = spins["timestamps"][j]
        want = rim.predict(ts[: len(ts) - M], spins["ss"][j])
        assert r.prediction == want
        assert r.t_strike_pred == ts[len(ts) - 1 - M] + want.t_f
        assert r.t_strike == spins["t_strikes"][j]
        assert r.fit["eta"] == rim.params.dep_params.eta
        assert r.fit["a"] == rim.params.ball_params.a


def test_worker_count_does_not_change_results(spins):
    hold = [i for i, t in zip(spins["ids"], spins["t_strikes"]) if t is not None][:6]
    serial = leave_one_out(*args(spins), hold_out=hold, M=M, Y=Y, config=CFG, workers=1)
    pooled = leave_one_out(*args(spins), hold_out=hold, M=M, Y=Y, config=CFG, workers=3)
    assert serial == pooled


def test_default_hold_out_is_every_spin_that_calibrates(spins):
    short = dict(spins)
    short["timestamps"] = list(spins["timestamps"])
    short["timestamps"][0] = spins["timestamps"][0][: M + Y]     # too few to anchor
    got = leave_one_out(*args(short), M=M, Y=Y, config=CFG, workers=1)
    expected = [i for j, (i, t) in enumerate(zip(spins["ids"], spins["t_strikes"]))
                if t is not None and j != 0]
    assert [r.id for r in got] == expected

    s = summary(got)
    assert s["n"] == len(expected) and s["n_errors"] == 0
    assert 0.0 <= s["hit_rate"] <= 1.0 and math.isfinite(s["median_abs_dt"])


def test_refusals_are_reported_not_raised(spins):
    # holding out every anticlockwise strike leaves that sense uncalibrated
    ccw = [i for i, t, s in zip(spins["ids"], spins["t_strikes"], spins["ss"])
           if t is not None and s < 0]
    got = cross_validate(*args(spins), [ccw], M=M, Y=Y, config=CFG, workers=1)
    assert all(r.prediction is None and "no strike was calibrated" in r.error for r in got)


def test_bad_folds_are_rejected(spins):
    with pytest.raises(ValueError, match="unknown held-out"):
        cross_validate(*args(spins), [["nope"]], config=CFG, workers=1)
