"""The compiled kernels are the numpy reference, per spin.

:mod:`strike_fit._kernels` exists only for speed. These pin it to
:mod:`strike_fit.model` and :func:`strike_fit.calibrate.evaluate` at rounding
level, across both senses, the flagged spins and the channel switches, so the
speed work cannot quietly change what the calibration fits.
"""

import math

import numpy as np
import pytest

from strike_fit import _kernels, model
from strike_fit.calibrate import CalibrationConfig, StrikeObservations, calibrate, evaluate
from strike_fit.conventions import TWO_PI

A, B = 0.0225, math.sqrt(10.0 / 3.0)


def observations(n=60, seed=0):
    """Spins spread wide enough to hit every exit flag somewhere on the grid."""
    rng = np.random.default_rng(seed)
    omega0 = rng.uniform(2.0, 9.0, n)
    omega0[:3] = [np.nan, 0.5 * B, B]            # no speed, sub-critical, c1 == 0
    To = np.array([_to_from_omega0(w) for w in omega0])
    s = np.where(rng.random(n) < 0.5, 1, -1)
    theta_s = rng.uniform(10.0, 40.0, n)
    t_s = rng.uniform(2.0, 9.0, n)
    return StrikeObservations(To, theta_s, t_s, s, [f"x{j}" for j in range(n)])


def _to_from_omega0(omega0):
    if not np.isfinite(omega0) or omega0 <= B:
        return 1e3                                  # omega0_from_first_lap gives NaN
    return float(model.lap_times(1, A, B, omega0))


def kernel_args(cfg):
    lo_a, hi_a = cfg.A0_bounds if cfg.A0_bounds is not None else (0.0, 0.0)
    lo_c, hi_c = cfg.C0_bounds if cfg.C0_bounds is not None else (0.0, 0.0)
    return (cfg.use_angle_channel, cfg.use_time_channel,
            lo_a, hi_a, cfg.A0_bounds is not None, lo_c, hi_c, cfg.C0_bounds is not None)


@pytest.mark.parametrize("eta", [0.0, 0.3, 1.0, 3.0])
@pytest.mark.parametrize("s", [1.0, -1.0])
def test_exit_theta_and_fall_time_match_model(eta, s):
    rng = np.random.default_rng(int(eta * 10) + int(s > 0))
    omega0 = rng.uniform(1.0, 9.0, 400)
    for omega_sq in (3.0, 7.62, 11.0):
        phi = s * rng.uniform(0.0, TWO_PI, omega0.size)
        ref = model.predict(A, B, omega0, eta, phi, omega_sq)
        for j in range(omega0.size):
            c1 = omega0[j] ** 2 - B * B
            th = _kernels.exit_theta(A, B, c1, eta, phi[j], omega_sq)
            if np.isnan(ref.theta_f[j]):
                assert np.isnan(th)
                continue
            # numpy's vectorised sin/cos and libm's differ in the last digit
            assert th == pytest.approx(ref.theta_f[j], rel=1e-12, abs=1e-12)
            tf = _kernels.fall_time(th, A, B, c1, eta, phi[j])
            assert tf == pytest.approx(ref.t_f[j], rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("cfg", [
    CalibrationConfig(),
    CalibrationConfig(A0_bounds=None, C0_bounds=None),
    CalibrationConfig(use_time_channel=False),
    CalibrationConfig(use_angle_channel=False),
], ids=["default", "unbounded", "angle-only", "time-only"])
def test_objective_matches_evaluate(cfg):
    obs = observations()
    omega0 = model.omega0_from_first_lap(obs.To, A, B)
    s = obs.s.astype(float)
    rng = np.random.default_rng(3)
    for _ in range(150):
        delta, eta, omega_sq = rng.uniform(0, TWO_PI), rng.uniform(0, 1.5), rng.uniform(2.0, 12.0)
        s_theta, s_t = rng.uniform(0.05, 0.5), rng.uniform(0.01, 0.3)
        ref = evaluate(obs, A, B, delta, eta, omega_sq, s_theta, s_t, cfg)

        gap_theta, gap_t = np.empty(len(obs)), np.empty(len(obs))
        _kernels.gaps(omega0, s, obs.theta_s, obs.t_s, A, B, delta, eta, omega_sq, gap_theta, gap_t)
        S, A0_cw, A0_ccw, C0 = _kernels.score(gap_theta, gap_t, s, omega_sq, s_theta, s_t,
                                              *kernel_args(cfg))
        assert int(np.isnan(gap_theta).sum()) == ref.n_failed
        assert S == pytest.approx(ref.S, rel=1e-12)
        np.testing.assert_allclose([A0_cw, A0_ccw, C0], [ref.A0[1], ref.A0[-1], ref.C0],
                                   rtol=1e-12, atol=1e-12, equal_nan=True)
        assert _kernels.objective(omega0, s, obs.theta_s, obs.t_s, A, B, delta, eta, omega_sq,
                                  s_theta, s_t, *kernel_args(cfg)) == S


def test_grid_matches_cellwise_evaluate():
    obs = observations(n=40, seed=1)
    cfg = CalibrationConfig(delta_grid=np.deg2rad(np.arange(0.0, 360.0, 60.0)),
                            eta_grid=np.array([0.0, 0.4, 0.9]),
                            omega_sq_grid=np.array([6.0, 8.0]))
    omega0 = model.omega0_from_first_lap(obs.To, A, B)
    s = obs.s.astype(float)
    cached = _kernels.grid_gaps(omega0, s, obs.theta_s, obs.t_s, A, B,
                                cfg.delta_grid, cfg.eta_grid, cfg.omega_sq_grid)
    for s_theta, s_t in ((0.227, 0.15), (0.6, 0.02)):
        got = _kernels.grid_scores(cached, s, cfg.omega_sq_grid, s_theta, s_t, *kernel_args(cfg))
        for (i, j, k), S in np.ndenumerate(got):
            ref = evaluate(obs, A, B, cfg.delta_grid[i], cfg.eta_grid[j], cfg.omega_sq_grid[k],
                           s_theta, s_t, cfg)
            assert S == pytest.approx(ref.S, rel=1e-12)


def test_calibration_is_deterministic():
    """Same inputs, same answer, whatever the thread schedule of the grid."""
    obs = observations(n=50, seed=4)
    obs.theta_s[:] = np.abs(obs.theta_s)
    cfg = CalibrationConfig(delta_grid=np.deg2rad(np.arange(0.0, 360.0, 45.0)),
                            eta_grid=np.array([0.0, 0.3, 0.6]),
                            omega_sq_grid=np.array([6.5, 7.5, 8.5]), n_starts=2)
    first = calibrate(obs, A, B, cfg).to_dict()
    assert calibrate(obs, A, B, cfg).to_dict() == first
