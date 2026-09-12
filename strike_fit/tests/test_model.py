import math

import numpy as np
import pytest

from strike_fit import model
from strike_fit.conventions import OMEGA_F_SQ, TWO_PI

A, B = 0.0225, math.sqrt(10.0 / 3.0)


def tilted_lap_time_quadrature(c1, a, b, eta, phi, k0=0, n=20000):
    """Exact time for one revolution [2*pi*k0, 2*pi*(k0+1)] under the tilted law."""
    th = np.linspace(TWO_PI * k0, TWO_PI * (k0 + 1), 2 * n + 1)
    om2 = c1 * np.exp(-2 * a * th) + b * b + eta * (np.cos(th + phi) - 2 * a * np.sin(th + phi))
    f = 1 / np.sqrt(om2)
    h = TWO_PI / (2 * n)
    return h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())


def test_level_time_is_inverse_of_speed_law():
    # dt/dtheta must equal 1/Omega along the level law
    om0 = 6.0
    th = np.linspace(0, 40, 40001)
    t = model.level_time(th, A, B, om0)
    dt = np.gradient(t, th, edge_order=2)
    om = np.sqrt(model.level_omega_sq(th, A, B, om0))
    assert np.allclose(dt, 1 / om, rtol=1e-6)
    assert model.level_time(0.0, A, B, om0) == pytest.approx(0.0, abs=1e-12)


def test_omega0_from_first_lap_roundtrip():
    om0 = np.array([4.5, 5.5, 7.5])
    T0 = model.lap_times(1, A, B, om0[:, None])[:, 0]
    assert np.allclose(model.omega0_from_first_lap(T0, A, B), om0, rtol=1e-10)


@pytest.mark.parametrize("eta", [0.1, 0.2, 0.4])
@pytest.mark.parametrize("phi", [0.0, 1.0, 2.2, -1.3])
def test_p3_level_law_reproduces_tilted_lap_times(eta, phi):
    """P3 / D10: the level law with the same c1 gives the tilted lap times to ~1 ms."""
    om0 = 6.0
    c1 = model.c1_from_omega0(om0, B)
    n_laps = int(model.level_exit_angle(A, B, om0) // TWO_PI)   # laps before exit
    assert n_laps >= 6
    worst = 0.0
    for k0 in range(n_laps):
        exact = tilted_lap_time_quadrature(c1, A, B, eta, phi, k0)
        level = model.lap_times(k0 + 1, A, B, om0) - model.lap_times(k0, A, B, om0)
        worst = max(worst, abs(exact - level))
    # first-order cancellation leaves the paper's second-order term and the
    # slow drift of the boundary terms; both are well under click noise (30 ms)
    assert worst < 1.5e-3 + 6e-3 * eta


def test_exit_angle_eta_zero_matches_closed_form():
    om0 = np.linspace(4.5, 7.5, 7)
    ex = model.exit_angle(A, B, om0, 0.0, 0.0)
    assert np.all(ex.flags == 0)
    assert np.allclose(ex.theta_f, model.level_exit_angle(A, B, om0), rtol=1e-10)
    tf = model.fall_time(ex.theta_f, A, B, om0, 0.0, 0.0)
    assert np.allclose(tf, model.level_time(ex.theta_f, A, B, om0), rtol=1e-12)


@pytest.mark.parametrize("eta", [0.05, 0.2, 0.4, 0.6])
def test_exit_angle_is_first_root(eta):
    rng = np.random.default_rng(1)
    om0 = rng.uniform(4.5, 7.5, 300)
    phi = rng.uniform(-math.pi, math.pi, 300)
    ex = model.exit_angle(A, B, om0, eta, phi)
    ok = ex.flags == 0
    assert ok.mean() > 0.95
    g_at = model.g_exit(ex.theta_f[ok], A, B, ex.c1[ok], eta, phi[ok])
    assert np.max(np.abs(g_at)) < 1e-9
    # no earlier crossing on a fine grid, except a dip narrower than one march
    # step (which the march cannot resolve and which is physically marginal)
    for i in np.nonzero(ok)[0][:60]:
        th = np.linspace(0, ex.theta_f[i] * (1 - 1e-6), 8000)
        g = model.g_exit(th, A, B, ex.c1[i], eta, phi[i])
        neg = g <= 0
        if neg.any():
            runs = np.diff(np.flatnonzero(np.diff(np.r_[0, neg.astype(int), 0])))[::2]
            assert runs.max() * (th[1] - th[0]) < model.MARCH_STEP


def test_exit_flags():
    # sub-critical at the anchor
    ex = model.exit_angle(A, B, np.array([1.0]), 0.2, 0.0)
    assert ex.flags[0] & model.FLAG_NO_C1
    assert np.isnan(ex.theta_f[0])
    # at threshold already
    om_thr = math.sqrt(OMEGA_F_SQ) - 1e-3
    ex = model.exit_angle(A, B, np.array([om_thr]), 0.0, 0.0)
    assert ex.flags[0] & model.FLAG_G0_NONPOS
    # never exits: threshold below the asymptote b^2
    ex = model.exit_angle(A, B, np.array([6.0]), 0.0, 0.0, omega_f_sq=B * B - 0.1)
    assert ex.flags[0] & model.FLAG_NO_EXIT


@pytest.mark.parametrize("eta", [0.1, 0.2, 0.4])
def test_fall_time_matches_quadrature(eta):
    rng = np.random.default_rng(2)
    om0 = rng.uniform(4.5, 7.5, 60)
    phi = rng.uniform(-math.pi, math.pi, 60)
    ex = model.exit_angle(A, B, om0, eta, phi)
    ok = ex.flags == 0
    tf = model.fall_time(ex.theta_f[ok], A, B, om0[ok], eta, phi[ok])
    tq = np.array([model.fall_time_quadrature(t, A, B, o, eta, p)
                   for t, o, p in zip(ex.theta_f[ok], om0[ok], phi[ok])])
    err = tf - tq
    # paper's bound: second-order term < 0.02 eta^2 s, plus the dropped
    # boundary term ~ (eta/2)(sin phi + 2a cos phi)/Omega0^3
    bound = 0.02 * eta ** 2 + 0.5 * eta * 1.05 / 4.5 ** 3 + 2e-3
    assert np.max(np.abs(err)) < bound
    # the error is common-mode to the level C0 absorbs: its spread is tiny
    assert np.std(err) < 2e-3


def test_predict_shapes_and_nan_propagation():
    om0 = np.array([6.0, 1.0, 5.0])
    p = model.predict(A, B, om0, 0.2, np.array([0.5, 0.5, 0.5]))
    assert p.theta_f.shape == (3,) and p.t_f.shape == (3,)
    assert np.isnan(p.theta_f[1]) and np.isnan(p.t_f[1])
    assert np.isfinite(p.t_f[[0, 2]]).all()


# ----------------------------------------------------------------------------
# Cross-check against the C++ bindings (skipped when `stator` is unavailable)
# ----------------------------------------------------------------------------

try:
    import stator  # noqa: F401
    HAVE_STATOR = True
except Exception:  # pragma: no cover
    HAVE_STATOR = False


@pytest.mark.skipif(not HAVE_STATOR, reason="C++ bindings not importable in this environment")
@pytest.mark.parametrize("s", [1.0, -1.0])
@pytest.mark.parametrize("delta", [0.0, 0.7, 2.9, 5.1])
@pytest.mark.parametrize("To", [0.9, 1.15, 1.4])
def test_matches_cpp_predict(s, delta, To):
    eta = 0.2
    params = stator.FitParams(stator.BallParams(A, B), stator.DepartureParams(delta, eta, OMEGA_F_SQ))
    th_cpp = stator.predict_theta(To, s, params)
    om0 = model.omega0_from_first_lap(To, A, B)
    phi = s * delta
    ex = model.exit_angle(A, B, np.array([om0]), eta, np.array([phi]))
    if th_cpp is None:
        assert ex.flags[0] != 0
        return
    assert ex.flags[0] == 0
    assert ex.theta_f[0] == pytest.approx(th_cpp, abs=1e-6)
    tf_cpp = stator.predict_tf(To, th_cpp, s, params)
    tf = model.fall_time(ex.theta_f[0], A, B, om0, eta, phi)
    assert tf == pytest.approx(tf_cpp, abs=1e-6)
