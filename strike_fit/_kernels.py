"""Compiled inner loops of the strike calibration.

The calibration objective is evaluated about ten thousand times per fit. The
vectorised numpy forms in :mod:`strike_fit.model` spend almost all of that
time dispatching small-array calls, not computing, so the same arithmetic is
repeated here per spin under numba. Nothing in this module is a new model:

* :func:`exit_theta` is :func:`strike_fit.model.exit_angle` for one spin -- the
  same march start, step, revolution cap and bracket rules -- and
  :func:`fall_time` is :func:`strike_fit.model.fall_time`. The one difference
  is how the bracketed root is polished: safeguarded Newton, as
  ``core/rtsafe.h`` does, instead of sixty bisections. Both converge to
  rounding on the same bracket; the roots agree to about 1e-13 rad.
* :func:`score` is the objective of :func:`strike_fit.calibrate.evaluate`:
  projected medians for ``A0`` per direction and ``C0``, L1 sums, and the
  failure penalty.

``tests/test_kernels.py`` pins both against the numpy forms. No ``fastmath``:
the kernels must reproduce the reference to rounding, not approximately.

The forward model does not depend on the residual scales, so the grid is split
in two: :func:`grid_gaps` solves every cell once, and :func:`grid_scores`
scores the cached gaps under any ``(s_theta, s_t)``. The rescaled second pass
of the calibration therefore costs only the scoring.
"""

from __future__ import annotations

import math

import numba as nb
import numpy as np

from .conventions import TWO_PI
from .model import MARCH_STEP, MAX_REVOLUTIONS

N_MARCH_STEPS = int(math.ceil(MAX_REVOLUTIONS * TWO_PI / MARCH_STEP))
ROOT_MAXIT = 100
EPS = np.finfo(np.float64).eps


@nb.njit(cache=True, inline="always")
def _g(theta, a, c1, eta, phi, k, K):
    """eq. 37 left-hand side, ``K = omega_f_sq - b^2``."""
    return (c1 * math.exp(-2.0 * a * theta)
            + eta * (k * math.cos(theta + phi) - 2.0 * a * math.sin(theta + phi))
            - K)


@nb.njit(cache=True, inline="always")
def _g_prime(theta, a, c1, eta, phi, k):
    return (-2.0 * a * c1 * math.exp(-2.0 * a * theta)
            - eta * (k * math.sin(theta + phi) + 2.0 * a * math.cos(theta + phi)))


@nb.njit(cache=True)
def exit_theta(a, b, c1, eta, phi, omega_f_sq):
    """Smallest positive root of eq. 37 for one spin; NaN where
    :func:`strike_fit.model.exit_angle` flags the spin."""
    if not c1 > 0.0:
        return np.nan
    k = 1.0 + 0.5 * (4.0 * a * a + 1.0)
    K = omega_f_sq - b * b
    if _g(0.0, a, c1, eta, phi, k, K) <= 0.0:
        return np.nan
    M = abs(eta) * math.sqrt(k * k + 4.0 * a * a)
    if M + K <= 0.0:
        return np.nan

    theta_lo = -math.log((M + K) / c1) / (2.0 * a)
    if not math.isfinite(theta_lo):
        theta_lo = 0.0
    theta_lo = max(theta_lo, 0.0)

    lo = theta_lo
    if _g(lo, a, c1, eta, phi, k, K) <= 0.0:
        hi = lo
        lo = max(lo - MARCH_STEP, 0.0)
    else:
        hi = lo
        found = False
        for i in range(1, N_MARCH_STEPS + 1):
            hi = theta_lo + i * MARCH_STEP
            if _g(hi, a, c1, eta, phi, k, K) <= 0.0:
                found = True
                break
            lo = hi
        if not found:
            return np.nan

    # Safeguarded Newton on the bracket, as core/rtsafe.h: a Newton step when it
    # lands inside the bracket and is shrinking fast enough, a bisection
    # otherwise. g(lo) > 0 >= g(hi), so the negative end is hi.
    x_neg = hi
    x_pos = lo
    root = 0.5 * (lo + hi)
    dx_old = abs(hi - lo)
    dx = dx_old
    f = _g(root, a, c1, eta, phi, k, K)
    df = _g_prime(root, a, c1, eta, phi, k)
    for _ in range(ROOT_MAXIT):
        if (((root - x_pos) * df - f) * ((root - x_neg) * df - f) > 0.0
                or abs(2.0 * f) > abs(dx_old * df)):
            dx_old = dx
            dx = 0.5 * (x_pos - x_neg)
            root = x_neg + dx
            if x_neg == root:
                break
        else:
            dx_old = dx
            dx = f / df
            prev = root
            root -= dx
            if prev == root:
                break
        if abs(dx) < 2.0 * abs(root) * EPS:
            break
        f = _g(root, a, c1, eta, phi, k, K)
        df = _g_prime(root, a, c1, eta, phi, k)
        if f < 0.0:
            x_neg = root
        else:
            x_pos = root
    return root


@nb.njit(cache=True)
def fall_time(theta_f, a, b, c1, eta, phi):
    """eq. 39 for one spin, as :func:`strike_fit.model.fall_time`."""
    c0 = -math.asinh(b / math.sqrt(c1))
    level = (c0 - math.asinh(math.sinh(c0) * math.exp(a * theta_f))) / (a * b)
    tilt = (0.5 * eta * (math.sin(theta_f + phi) + 2.0 * a * math.cos(theta_f + phi))
            / (c1 * math.exp(-2.0 * a * theta_f) + b * b) ** 1.5)
    return level - tilt


@nb.njit(cache=True)
def gaps(omega0, s, theta_s, t_s, a, b, delta, eta, omega_sq, gap_theta, gap_t):
    """Fill the observed-minus-model gaps for one parameter cell. NaN marks a
    spin the model cannot exit, exactly the spins ``evaluate`` counts as failed."""
    for j in range(omega0.size):
        gap_theta[j] = np.nan
        gap_t[j] = np.nan
        if not math.isfinite(omega0[j]):
            continue
        c1 = omega0[j] * omega0[j] - b * b
        phi = s[j] * delta
        th = exit_theta(a, b, c1, eta, phi, omega_sq)
        if not math.isfinite(th):
            continue
        tf = fall_time(th, a, b, c1, eta, phi)
        if not math.isfinite(tf):
            continue
        gap_theta[j] = theta_s[j] - th
        gap_t[j] = t_s[j] - tf


@nb.njit(cache=True)
def _bounded_median(x, lo, hi, bounded):
    if x.size == 0:
        return np.nan
    m = np.median(x)
    if bounded:
        m = min(max(m, lo), hi)
    return m


@nb.njit(cache=True)
def score(gap_theta, gap_t, s, omega_sq, s_theta, s_t, use_angle, use_time,
          A0_lo, A0_hi, A0_bounded, C0_lo, C0_hi, C0_bounded):
    """Objective and inner offsets from one cell's gaps.

    Returns ``(S, A0_cw, A0_ccw, C0)``, matching ``calibrate.evaluate``.
    """
    n = gap_theta.size
    valid = np.isfinite(gap_theta)
    n_failed = n - int(valid.sum())
    A0_cw = _bounded_median(gap_theta[valid & (s > 0.0)], A0_lo, A0_hi, A0_bounded)
    A0_ccw = _bounded_median(gap_theta[valid & (s < 0.0)], A0_lo, A0_hi, A0_bounded)
    C0 = _bounded_median(gap_t[valid], C0_lo, C0_hi, C0_bounded)

    sum_theta = 0.0
    sum_t = 0.0
    for j in range(n):
        if not valid[j]:
            continue
        A0 = A0_cw if s[j] > 0.0 else A0_ccw
        if math.isfinite(A0):
            sum_theta += abs(gap_theta[j] - A0)
        sum_t += abs(gap_t[j] - C0)

    S = 0.0
    penalty = 0.0
    if use_angle:
        S += sum_theta / s_theta
        penalty += math.pi / s_theta
    if use_time:
        S += sum_t / s_t
        penalty += TWO_PI / math.sqrt(max(omega_sq, 1e-6)) / s_t
    S += penalty * n_failed
    return S, A0_cw, A0_ccw, C0


@nb.njit(cache=True)
def objective(omega0, s, theta_s, t_s, a, b, delta, eta, omega_sq, s_theta, s_t,
              use_angle, use_time, A0_lo, A0_hi, A0_bounded, C0_lo, C0_hi, C0_bounded):
    """``calibrate.evaluate(...).S`` at one cell, for the polish step."""
    gap_theta = np.empty(omega0.size)
    gap_t = np.empty(omega0.size)
    gaps(omega0, s, theta_s, t_s, a, b, delta, eta, omega_sq, gap_theta, gap_t)
    return score(gap_theta, gap_t, s, omega_sq, s_theta, s_t, use_angle, use_time,
                 A0_lo, A0_hi, A0_bounded, C0_lo, C0_hi, C0_bounded)[0]


@nb.njit(cache=True, parallel=True)
def grid_gaps(omega0, s, theta_s, t_s, a, b, delta_grid, eta_grid, omega_sq_grid):
    """Gaps for every grid cell, shape ``(n_delta, n_eta, n_omega_sq, 2, n_spins)``."""
    n = omega0.size
    out = np.empty((delta_grid.size, eta_grid.size, omega_sq_grid.size, 2, n))
    for i in nb.prange(delta_grid.size):
        for j in range(eta_grid.size):
            for k in range(omega_sq_grid.size):
                gaps(omega0, s, theta_s, t_s, a, b, delta_grid[i], eta_grid[j],
                     omega_sq_grid[k], out[i, j, k, 0], out[i, j, k, 1])
    return out


@nb.njit(cache=True, parallel=True)
def grid_scores(cached_gaps, s, omega_sq_grid, s_theta, s_t, use_angle, use_time,
                A0_lo, A0_hi, A0_bounded, C0_lo, C0_hi, C0_bounded):
    """Objective for every cached cell under the given residual scales."""
    n_d, n_e, n_w = cached_gaps.shape[0], cached_gaps.shape[1], cached_gaps.shape[2]
    out = np.empty((n_d, n_e, n_w))
    for i in nb.prange(n_d):
        for j in range(n_e):
            for k in range(n_w):
                out[i, j, k] = score(cached_gaps[i, j, k, 0], cached_gaps[i, j, k, 1], s,
                                     omega_sq_grid[k], s_theta, s_t, use_angle, use_time,
                                     A0_lo, A0_hi, A0_bounded, C0_lo, C0_hi, C0_bounded)[0]
    return out
