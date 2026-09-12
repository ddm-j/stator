"""Closed-form rim model (Eichberger 2004), vectorised over spins.

All angles are travel-frame (see conventions.py). ``theta = 0`` and ``t = 0``
at the anchor click; ``omega0`` is the ball's angular speed at the anchor
under the *level* backbone (Stage A). The tilt enters only through
``(eta, phi)`` with ``phi = s * delta``.

Equation numbers refer to the paper.

* eq. 35 -- level lap-time law ``t(theta)``; used for the Stage-A model and
  as the backbone of the fall time.
* eq. 37 -- exit condition ``g(theta) = 0`` on a tilted wheel with the
  observation point at phase ``phi`` from the low point.
* eq. 38/39 -- fall time, exact quadrature and the paper's closed-form
  approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

from .conventions import OMEGA_F_SQ, TWO_PI, kappa

# Exit-search controls. The brief suggests a pi/8 march; predict.h uses
# 2*pi/64 and at eta ~ 0.4 the coarser step can skip a narrow genuine crossing,
# so we match the C++ (4x the cost, still trivial vectorised).
MARCH_STEP = TWO_PI / 64.0
MAX_REVOLUTIONS = 30
BISECTION_ITERS = 60

# Per-spin flags, as bit values so they can be OR-ed and stored compactly.
FLAG_OK = 0
FLAG_NO_C1 = 1          # c1 <= 0: sub-critical at the anchor under this tilt
FLAG_NO_EXIT = 2        # no sign change of g within MAX_REVOLUTIONS
FLAG_G0_NONPOS = 4      # g(0) <= 0: already at/below threshold at the anchor

FLAG_NAMES = {
    FLAG_NO_C1: "NO_C1",
    FLAG_NO_EXIT: "NO_EXIT",
    FLAG_G0_NONPOS: "G0_NONPOS",
}


def flag_names(flag: int) -> list[str]:
    return [name for bit, name in FLAG_NAMES.items() if flag & bit]


# ----------------------------------------------------------------------------
# Level backbone (Stage A)
# ----------------------------------------------------------------------------

def arcoth(x):
    return 0.5 * np.log((x + 1.0) / (x - 1.0))


def c0_from_omega0(omega0, b):
    """``c0 = -arcoth(omega0 / b)`` (< 0 for ``omega0 > b``)."""
    return -arcoth(np.asarray(omega0, dtype=float) / b)


def c0_from_c1(c1, b):
    """``c0 = -asinh(b / sqrt(c1))`` -- the same relation written through ``c1``.

    Identical to ``c0_from_omega0`` for ``c1 = omega0^2 - b^2`` (and to
    ``-acoth(x)`` with ``c1 = b^2 (x^2 - 1)`` in predict.h).
    """
    return -np.arcsinh(b / np.sqrt(np.asarray(c1, dtype=float)))


def omega0_from_first_lap(T0, a, b):
    """Level-model speed at the anchor given the first lap duration (eq. 14/15).

    ``x = (e^{2 pi a} - cosh(ab T0)) / sinh(ab T0)``, ``c1 = b^2 (x^2 - 1)``,
    ``omega0 = sqrt(c1 + b^2) = b |x|``. Returns NaN where ``x <= 1``.
    """
    T0 = np.asarray(T0, dtype=float)
    with np.errstate(all="ignore"):
        x = (np.exp(TWO_PI * a) - np.cosh(a * b * T0)) / np.sinh(a * b * T0)
        out = np.where(x > 1.0, b * x, np.nan)
    return out if out.ndim else float(out)


def level_time(theta, a, b, omega0):
    """eq. 35: time to reach travel angle ``theta`` on a level wheel.

    Broadcasts ``theta`` against ``omega0``; pass ``theta`` with shape
    ``(n_spins, n_k)`` and ``omega0`` with shape ``(n_spins, 1)`` for a table.
    """
    c0 = c0_from_omega0(omega0, b)
    return (c0 - np.arcsinh(np.sinh(c0) * np.exp(a * np.asarray(theta, dtype=float)))) / (a * b)


def lap_times(k, a, b, omega0):
    """eq. 35 at complete revolutions: ``t(k * 2 pi)``. Tilt-independent (P3)."""
    return level_time(TWO_PI * np.asarray(k, dtype=float), a, b, omega0)


def level_omega_sq(theta, a, b, omega0):
    """``Omega^2 = c1 e^{-2 a theta} + b^2`` with ``c1 = omega0^2 - b^2``."""
    c1 = np.asarray(omega0, dtype=float) ** 2 - b * b
    return c1 * np.exp(-2.0 * a * np.asarray(theta, dtype=float)) + b * b


# ----------------------------------------------------------------------------
# Tilted exit condition (eq. 37) and fall time (eq. 38/39)
# ----------------------------------------------------------------------------

def c1_from_omega0(omega0, b):
    """``c1 = omega0^2 - b^2`` where ``omega0`` is the Stage-A (level-fit) speed.

    Stage A fits lap clicks with the level law, and by P3 the level law with
    constant ``c1`` reproduces the tilted wheel's lap times for the *same*
    ``c1`` (paper section 4.4). So the level-fit ``omega0`` is the
    level-equivalent speed and ``omega0^2 - b^2`` is already the tilted law's
    constant; nothing is subtracted. This matches ``predict.h``
    (``c1 = b^2 (x^2 - 1)`` straight from ``To``) and departs from the brief's
    section 5.1, which subtracts ``eta (cos phi - 2a sin phi)`` a second time
    (PLAN.md D10). The ball's true instantaneous speed at the anchor is
    ``sqrt(c1 + b^2 + eta (cos phi - 2a sin phi))`` and is never needed.
    """
    return np.asarray(omega0, dtype=float) ** 2 - b * b


def g_exit(theta, a, b, c1, eta, phi, omega_f_sq=OMEGA_F_SQ):
    """eq. 37 left-hand side. Positive while the ball is supercritical."""
    k = kappa(a)
    th = np.asarray(theta, dtype=float)
    return (c1 * np.exp(-2.0 * a * th)
            + eta * (k * np.cos(th + phi) - 2.0 * a * np.sin(th + phi))
            + b * b - omega_f_sq)


def g_exit_prime(theta, a, b, c1, eta, phi):
    k = kappa(a)
    th = np.asarray(theta, dtype=float)
    return (-2.0 * a * c1 * np.exp(-2.0 * a * th)
            - eta * (k * np.sin(th + phi) + 2.0 * a * np.cos(th + phi)))


def level_exit_angle(a, b, omega0, omega_f_sq=OMEGA_F_SQ):
    """Closed-form exit angle with no tilt: ``c1 e^{-2 a theta} = omega_f^2 - b^2``."""
    c1 = np.asarray(omega0, dtype=float) ** 2 - b * b
    return -np.log((omega_f_sq - b * b) / c1) / (2.0 * a)


@dataclass
class ExitResult:
    theta_f: np.ndarray     # NaN where flagged
    flags: np.ndarray       # int bit flags per spin
    g_min: np.ndarray       # minimum of g seen on the march before the accepted bracket
    c1: np.ndarray


def exit_angle(a, b, omega0, eta, phi, omega_f_sq=OMEGA_F_SQ,
               step=MARCH_STEP, max_revolutions=MAX_REVOLUTIONS) -> ExitResult:
    """Smallest positive root of eq. 37 for every spin (vectorised).

    March from ``theta = 0`` in steps of ``step`` until the first sign change
    of ``g``, then bisect the bracket (``BISECTION_ITERS`` halvings, ~1e-17
    of the bracket) and finish with two Newton steps. A dip that grazes zero
    without crossing produces no sign change and is stepped over; its depth is
    reported in ``g_min`` (per-spin ``marginal_exit`` diagnostic).
    """
    omega0 = np.atleast_1d(np.asarray(omega0, dtype=float))
    phi = np.broadcast_to(np.asarray(phi, dtype=float), omega0.shape).copy()
    n = omega0.size

    c1 = c1_from_omega0(omega0, b)
    flags = np.zeros(n, dtype=int)
    theta_f = np.full(n, np.nan)
    g_min = np.full(n, np.inf)

    ok = c1 > 0.0
    flags[~ok] |= FLAG_NO_C1

    g0 = g_exit(0.0, a, b, c1, eta, phi, omega_f_sq)
    bad0 = ok & (g0 <= 0.0)
    flags[bad0] |= FLAG_G0_NONPOS
    ok &= ~bad0
    if not ok.any():
        return ExitResult(theta_f, flags, g_min, c1)

    # No exit is possible while c1 e^{-2a theta} > omega_f^2 - b^2 + M, where M
    # bounds the tilt term; start the march there (as predict.h does) instead
    # of at theta = 0. g stays positive on [0, theta_lo] by construction.
    k = kappa(a)
    M = abs(eta) * math.sqrt(k * k + 4.0 * a * a)
    K = omega_f_sq - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        theta_lo = np.where(ok & (M + K > 0.0), -np.log((M + K) / c1) / (2.0 * a), 0.0)
    theta_lo = np.where(np.isfinite(theta_lo), np.maximum(theta_lo, 0.0), 0.0)
    if M + K <= 0.0:
        flags[ok] |= FLAG_NO_EXIT
        return ExitResult(theta_f, flags, g_min, c1)

    lo = theta_lo.copy()
    g_lo = g_exit(lo, a, b, c1, eta, phi, omega_f_sq)
    # tangency / already-crossing at theta_lo (can only happen within one step of a true root)
    cross0 = ok & (g_lo <= 0.0)
    found = np.zeros(n, dtype=bool)
    hi_out = np.zeros(n)
    n_steps = int(np.ceil(max_revolutions * TWO_PI / step))
    active = ok & ~cross0
    # spins crossing exactly at theta_lo: bracket [max(0, lo - step), lo]
    if cross0.any():
        hi_out[cross0] = lo[cross0]
        lo[cross0] = np.maximum(lo[cross0] - step, 0.0)
        found |= cross0
    for i in range(1, n_steps + 1):
        if not active.any():
            break
        hi = theta_lo + i * step
        g_hi = g_exit(hi, a, b, c1, eta, phi, omega_f_sq)
        cross = active & (g_hi <= 0.0)
        hi_out[cross] = hi[cross]
        found |= cross
        active &= ~cross
        g_min[active] = np.minimum(g_min[active], g_hi[active])
        lo[active] = hi[active]
        g_lo[active] = g_hi[active]

    flags[ok & ~found] |= FLAG_NO_EXIT
    if not found.any():
        return ExitResult(theta_f, flags, g_min, c1)

    # Bisection on the bracketed spins
    idx = np.nonzero(found)[0]
    lo_b, hi_b = lo[idx], hi_out[idx]
    c1_b, phi_b = c1[idx], phi[idx]
    for _ in range(BISECTION_ITERS):
        mid = 0.5 * (lo_b + hi_b)
        g_mid = g_exit(mid, a, b, c1_b, eta, phi_b, omega_f_sq)
        pos = g_mid > 0.0
        lo_b = np.where(pos, mid, lo_b)
        hi_b = np.where(pos, hi_b, mid)
    root = 0.5 * (lo_b + hi_b)
    for _ in range(2):
        gp = g_exit_prime(root, a, b, c1_b, eta, phi_b)
        stepn = np.where(gp != 0.0, g_exit(root, a, b, c1_b, eta, phi_b, omega_f_sq) / gp, 0.0)
        cand = root - stepn
        inside = (cand >= lo[idx]) & (cand <= hi_out[idx])
        root = np.where(inside, cand, root)
    theta_f[idx] = root
    g_min[~found] = np.nan
    return ExitResult(theta_f, flags, g_min, c1)


def fall_time(theta_f, a, b, omega0, eta, phi):
    """eq. 39: time from the anchor to the exit angle ``theta_f``.

    The eq. 35 term with ``c0`` from ``c1 = omega0^2 - b^2`` plus the
    first-order tilt correction. Identical to ``predict_tf`` in
    ``stator/physics/predict.h`` (see ``c1_from_omega0`` for why there is only
    one backbone). The paper drops the lower-limit boundary term of its
    integration by parts, ``-(eta/2)(sin phi + 2a cos phi)/(c1 + b^2)^{3/2}``;
    it is sub-millisecond at first-click speeds and common-mode across spins,
    so ``C0`` absorbs it (tests quantify it against eq. 38 quadrature).
    """
    theta_f = np.asarray(theta_f, dtype=float)
    c1 = c1_from_omega0(omega0, b)
    c0 = c0_from_c1(c1, b)
    level = (c0 - np.arcsinh(np.sinh(c0) * np.exp(a * theta_f))) / (a * b)
    tilt = (0.5 * eta * (np.sin(theta_f + phi) + 2.0 * a * np.cos(theta_f + phi))
            / (c1 * np.exp(-2.0 * a * theta_f) + b * b) ** 1.5)
    return level - tilt


def fall_time_quadrature(theta_f, a, b, omega0, eta, phi, n=4000):
    """eq. 38 by composite Simpson quadrature. Reference for tests only."""
    theta_f = float(theta_f)
    c1 = float(c1_from_omega0(omega0, b))
    th = np.linspace(0.0, theta_f, 2 * n + 1)
    om2 = c1 * np.exp(-2.0 * a * th) + b * b + eta * (np.cos(th + phi) - 2.0 * a * np.sin(th + phi))
    f = 1.0 / np.sqrt(om2)
    h = theta_f / (2 * n)
    return float(h / 3.0 * (f[0] + f[-1] + 4.0 * f[1:-1:2].sum() + 2.0 * f[2:-1:2].sum()))


@dataclass
class Prediction:
    theta_f: np.ndarray
    t_f: np.ndarray
    flags: np.ndarray
    g_min: np.ndarray


def predict(a, b, omega0, eta, phi, omega_f_sq=OMEGA_F_SQ) -> Prediction:
    """Exit angle and fall time for every spin; NaN where flagged."""
    ex = exit_angle(a, b, omega0, eta, phi, omega_f_sq)
    tf = np.full_like(ex.theta_f, np.nan)
    ok = np.isfinite(ex.theta_f)
    if ok.any():
        om = np.atleast_1d(np.asarray(omega0, dtype=float))
        ph = np.broadcast_to(np.asarray(phi, dtype=float), om.shape)
        tf[ok] = fall_time(ex.theta_f[ok], a, b, om[ok], eta, ph[ok])
    return Prediction(ex.theta_f, tf, ex.flags, ex.g_min)
