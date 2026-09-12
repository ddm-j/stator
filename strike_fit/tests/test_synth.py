import math

import numpy as np
import pytest
from scipy.stats import binomtest, chisquare

from strike_fit import model, synth
from strike_fit.conventions import TWO_PI
from strike_fit.data import Dataset
from strike_fit.rim import deflector_travel_angle, strike_deflector


@pytest.fixture(scope="module")
def ds():
    return synth.generate(600, eta=0.2, seed=3)


def test_dataset_round_trips(ds, tmp_path):
    assert len(ds) == 600
    p = tmp_path / "syn.jsonl"
    ds.write_jsonl(p)
    ds2 = Dataset.read_jsonl(p)
    assert ds2.header.extra["truth"]["eta"] == 0.2
    assert synth.truth_from_header(ds2.header).A0 == {1: 0.5, -1: 0.7}


def test_spins_are_long_enough_for_the_lead_window(ds):
    n = np.array([len(r["clicks_s"]) for r in ds.records])
    assert n.min() >= 11          # M=4 plus Y=6 plus one
    assert n.max() < 30


def test_crossings_run_to_the_strike(ds):
    """Every crossing up to the strike is present, and no more."""
    for rec in ds.records:
        tr = rec["truth"]
        d, n_extra = strike_deflector(tr["theta_s"], 1 if rec["direction"] == "cw" else -1, 8)
        assert len(rec["clicks_s"]) == n_extra + 1 == tr["n_crossings"]
        if rec["strike"] is not None:
            assert rec["strike"]["deflector"] == d


def test_strike_angle_reconstructs_from_deflector_and_count(ds):
    """What the model reconstructs from the annotation brackets the true angle.

    The observer records a deflector and a crossing count; the model turns that
    back into an angle. The result must sit within one deflector spacing ahead
    of where the ball actually was, which is the quantisation the deflector
    channel is limited by.
    """
    spacing = TWO_PI / 8
    for rec in ds.records:
        if rec["strike"] is None:
            continue
        s = 1 if rec["direction"] == "cw" else -1
        n_cross = len(rec["clicks_s"])
        theta_model = (deflector_travel_angle(rec["strike"]["deflector"], s, 8)
                       + TWO_PI * (n_cross - 1))
        gap = theta_model - rec["truth"]["theta_s"]
        assert 0.0 <= gap < spacing + 1e-9


def test_off_rim_crossings_arrive_early(ds):
    """A crossing made after rim exit lands sooner than the rim law predicts."""
    truth = synth.truth_from_header(ds.header)
    early, on_rim = [], []
    for rec in ds.records:
        tr = rec["truth"]
        if tr["hover"]:
            continue
        k = np.arange(tr["n_crossings"])
        clean = model.lap_times(k, truth.a, truth.b, tr["omega0"])
        r = np.asarray(rec["clicks_s"]) - (clean - clean[0])
        past = TWO_PI * k > tr["theta_f"]
        early.extend(r[past].tolist())
        on_rim.extend(r[~past][1:].tolist())
    assert len(early) > 20
    assert np.median(early) < -0.05
    assert abs(np.median(on_rim)) < 0.01
    # each residual carries the click's own noise and the first click's, via re-zeroing
    assert np.std(on_rim) == pytest.approx(truth.sigma_click * math.sqrt(2), rel=0.15)


def test_clicks_are_valid_series(ds):
    for rec in ds.records:
        c = np.asarray(rec["clicks_s"])
        assert c[0] == 0.0
        assert np.all(np.diff(c) > 0)
        if rec["strike"] is not None:
            assert rec["strike"]["t_s"] > c[-1]


def test_injection_rates(ds):
    tr = ds.truth()
    n = len(tr)
    for key, p in (("hover", 0.08), ("missing_strike", 0.05)):
        cnt = sum(t[key] for t in tr)
        assert binomtest(cnt, n, p).pvalue > 1e-3, (key, cnt)
    for rec in ds.records:
        assert (rec["strike"] is None) == rec["truth"]["missing_strike"]


def test_hover_delays_time_and_adds_a_revolution():
    ds0 = synth.generate(300, eta=0.2, seed=5, p_hover=0.0, p_missing=0.0)
    ds1 = synth.generate(300, eta=0.2, seed=5, p_hover=1.0, p_missing=0.0)
    truth = synth.truth_from_header(ds0.header)
    for r0, r1 in zip(ds0.records, ds1.records):
        assert r0["truth"]["theta_f"] == r1["truth"]["theta_f"]
        assert r1["truth"]["theta_s"] == pytest.approx(r0["truth"]["theta_s"] + TWO_PI)
        assert r1["truth"]["n_crossings"] == r0["truth"]["n_crossings"] + 1
    lag = np.array([r1["strike"]["t_s"] - r0["strike"]["t_s"]
                    for r0, r1 in zip(ds0.records, ds1.records)])
    assert truth.T_rev == pytest.approx(2.276, abs=0.01)
    # one extra revolution of delay, less the crossing the re-zeroing absorbs
    assert abs(np.median(lag) - truth.T_rev) < 0.35


def test_oracle_is_good_but_not_perfect():
    ds = synth.generate(800, eta=0.2, seed=7)
    d_pred, t_pred = synth.oracle_predictions(ds)
    recs = ds.records
    has = np.array([r["strike"] is not None for r in recs])
    d_obs = np.array([r["strike"]["deflector"] if r["strike"] else -1 for r in recs])
    hit = (d_pred == d_obs)[has].mean()
    assert 0.75 < hit < 0.98        # limited by the descent-arc scatter
    hover = np.array([r["truth"]["hover"] for r in recs])
    t_obs = np.array([r["strike"]["t_s"] if r["strike"] else np.nan for r in recs])
    rt = (t_obs - t_pred)[has & ~hover]
    assert abs(np.median(rt)) < 0.01 and np.median(np.abs(rt)) < 0.03


def test_tilt_shows_in_the_deflector_histogram():
    ds0 = synth.generate(1500, eta=0.0, seed=8)
    for direction in ("cw", "ccw"):
        d = [r["strike"]["deflector"] for r in ds0.records
             if r["strike"] and r["direction"] == direction]
        assert chisquare(np.bincount(d, minlength=8)).pvalue > 1e-3
    ds2 = synth.generate(800, eta=0.2, seed=7)
    d = [r["strike"]["deflector"] for r in ds2.records
         if r["strike"] and r["direction"] == "cw"]
    assert chisquare(np.bincount(d, minlength=8)).pvalue < 1e-12


def test_reactive_mode_bias():
    ds_r = synth.generate(400, eta=0.2, seed=9, mode="reactive")
    assert ds_r.header.annotation_mode == "reactive"
    assert ds_r.header.extra["bias"] == synth.MODE_BIAS["reactive"]
    d_pred, t_pred = synth.oracle_predictions(ds_r)     # includes the bias
    recs = ds_r.records
    keep = np.array([r["strike"] is not None and not r["truth"]["hover"] for r in recs])
    t_obs = np.array([r["strike"]["t_s"] if r["strike"] else np.nan for r in recs])
    assert abs(np.median((t_obs - t_pred)[keep])) < 0.015
