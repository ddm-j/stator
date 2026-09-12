"""Strike calibration: the one piece of machinery this model does not share
with stator.

stator observes the departure angle. Its exit equation (paper eq. 37) is then
linear in ``(eta, omega_sq)`` at fixed ``delta``, which is why ``departure.h``
can solve those two with an L1 line fit inside a 1-D search over ``delta``.

This model observes the strike instead. A strike is a departure plus a descent,
and the descent contributes two unknown constants: an arc ``A0`` swept between
leaving the rim and reaching the deflector, and a time ``C0``. Those sit inside
the nonlinearity, so the same trick is not available and the search runs over
``(delta, eta, omega_sq)`` directly, with ``A0`` and ``C0`` solved exactly
inside every evaluation.

The inner solve is exact, not approximate. Both constants enter their residual
additively, and an L1 cost in an additive offset is minimised by the median, so
given ``(delta, eta, omega_sq)`` the best ``A0`` is the median of the angle gaps
and the best ``C0`` is the median of the time gaps. That leaves an outer search
of the same dimension as stator's.

Neither is a measurement of the descent. Both also soak up the systematic error
of the ``To`` estimator, which the exit angle amplifies by roughly ten radians
per rad/s: see the note beside ``A0_BOUNDS``.

The angle residual is **not** wrapped. The strike angle handed in by
:mod:`strike_fit.rim` carries its revolution count, because the annotation
convention counts every crossing up to the strike. Predicting the right
deflector one revolution early is a real error of 2*pi, and wrapping would hide
it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from . import model
from .conventions import OMEGA_F_SQ, TWO_PI, mad_scale  # noqa: F401

# ``A0`` and ``C0`` are effective offsets, not measurements of the descent.
# They also absorb the ``To`` estimator's systematic error, and the exit angle
# amplifies that badly: a 3% bias in the speed at the anchor moves the exit
# angle by well over a radian, and the amplification depends on the tilt phase,
# so it differs by direction. ``A0`` therefore needs room well beyond a
# physical descent arc.
#
# It still needs the one bound physics insists on. The ball leaves the rim
# before it strikes a deflector, so the arc is positive; and it is under a
# revolution, because every crossing up to the strike is tapped. Without the
# lower bound the search happily places rim exit *after* the strike, which fits
# in sample -- the offset absorbs anything -- and then fails out of sample,
# with different subsets landing a whole revolution apart.
#
# The time channel is not amplified the same way, so its bound can stay closer
# to physical. It is two-sided because the estimator's time bias has no
# preferred sign.
A0_BOUNDS = (0.0, TWO_PI)
C0_BOUNDS = (-1.0, 3.0)

S_T_INITIAL = {"mechanical": 0.03, "reactive": 0.15}
S_THETA_FLOOR = 0.02
S_T_FLOOR = 0.005


@dataclass
class CalibrationConfig:
    """Search grid and weighting. Defaults are the ones used on rc_208."""
    annotation_mode: str = "reactive"
    delta_grid: np.ndarray = field(
        default_factory=lambda: np.deg2rad(np.arange(0.0, 360.0, 15.0)))
    eta_grid: np.ndarray = field(
        default_factory=lambda: np.round(np.arange(0.0, 0.8001, 0.05), 6))
    omega_sq_grid: np.ndarray = field(
        default_factory=lambda: np.round(np.arange(6.0, 11.001, 0.5), 6))
    eta_max: float = 1.0
    # stator fits the exit threshold from departure angles, so by default this
    # fits it too, from strikes. Holding it at the paper's value instead makes
    # the descent constants individually identifiable, at the cost of assuming
    # a threshold: see the note in ``calibrate``.
    fit_omega_sq: bool = True
    omega_sq_fixed: float = OMEGA_F_SQ
    s_theta: float | None = None
    s_t: float | None = None
    rescale: bool = True
    rescale_clamp: float = 3.0
    n_starts: int = 4
    A0_bounds: tuple | None = A0_BOUNDS
    C0_bounds: tuple | None = C0_BOUNDS
    use_time_channel: bool = True
    use_angle_channel: bool = True


@dataclass
class StrikeObservations:
    """What the calibration sees, all referenced to each spin's anchor crossing."""
    To: np.ndarray          # lap duration at the anchor, stator's To estimator
    theta_s: np.ndarray     # travel from the anchor to the strike, unwrapped
    t_s: np.ndarray         # time from the anchor to the strike
    s: np.ndarray           # +1 clockwise, -1 anticlockwise
    ids: list = field(default_factory=list)

    def __len__(self):
        return int(self.To.size)


@dataclass
class Evaluation:
    S: float
    A0: dict
    C0: float
    r_theta: np.ndarray
    r_t: np.ndarray
    theta_f: np.ndarray
    t_f: np.ndarray
    flags: np.ndarray
    n_failed: int


def _bounded_median(x, bounds):
    """L1-optimal offset, projected onto its bound. NaN when there is nothing
    to estimate it from -- never a default, which would be indistinguishable
    from a fitted value downstream."""
    if x.size == 0:
        return float("nan")
    m = float(np.median(x))
    if bounds is None:
        return m
    return float(min(max(m, bounds[0]), bounds[1]))


def _inner_offsets(gap_theta, gap_t, s, valid, cfg):
    """Exact inner step: A0 per direction and C0, each a projected median."""
    A0 = {}
    for sgn in (1, -1):
        sel = valid & (s == sgn)
        A0[sgn] = _bounded_median(gap_theta[sel], cfg.A0_bounds)
    C0 = _bounded_median(gap_t[valid], cfg.C0_bounds)
    return A0, C0


def evaluate(obs: StrikeObservations, a, b, delta, eta, omega_sq, s_theta, s_t,
             cfg: CalibrationConfig) -> Evaluation:
    """Objective and the inner constants at one ``(delta, eta, omega_sq)``."""
    omega0 = model.omega0_from_first_lap(obs.To, a, b)
    pred = model.predict(a, b, omega0, eta, obs.s * delta, omega_sq)
    valid = np.isfinite(pred.theta_f) & np.isfinite(pred.t_f) & np.isfinite(omega0)

    gap_theta = np.where(valid, obs.theta_s - np.nan_to_num(pred.theta_f), 0.0)
    gap_t = np.where(valid, obs.t_s - np.nan_to_num(pred.t_f), 0.0)
    A0, C0 = _inner_offsets(gap_theta, gap_t, obs.s, valid, cfg)

    A0_arr = np.where(obs.s > 0, A0[1], A0[-1])
    r_theta = np.where(valid, gap_theta - A0_arr, np.nan)
    r_theta = np.where(np.isfinite(A0_arr), r_theta, np.nan)
    r_t = np.where(valid, gap_t - C0, np.nan)

    n_failed = int((~valid).sum())
    S = 0.0
    if cfg.use_angle_channel:
        S += float(np.nansum(np.abs(r_theta))) / s_theta
    if cfg.use_time_channel:
        S += float(np.nansum(np.abs(r_t))) / s_t
    # A spin the model cannot exit is not free: charge it half a revolution of
    # angle and one rim revolution of time, so a parameter set that simply
    # fails on hard spins cannot win.
    T_rev = TWO_PI / math.sqrt(max(omega_sq, 1e-6))
    penalty = 0.0
    if cfg.use_angle_channel:
        penalty += math.pi / s_theta
    if cfg.use_time_channel:
        penalty += T_rev / s_t
    S += penalty * n_failed
    return Evaluation(S, A0, C0, r_theta, r_t, pred.theta_f, pred.t_f, pred.flags, n_failed)


@dataclass
class StrikeCalibration:
    delta: float
    eta: float
    omega_sq: float
    A0: dict
    C0: float
    s_theta: float
    s_t: float
    objective: float
    n_spins: int
    n_failed: int
    n_by_direction: dict = field(default_factory=dict)
    eta_at_bound: bool = False      # tilt ran to its ceiling: the fit is straining
    r_theta: np.ndarray = None
    r_t: np.ndarray = None
    history: list = field(default_factory=list)

    def arc(self, s) -> float:
        return self.A0[1] if s > 0 else self.A0[-1]

    def to_dict(self) -> dict:
        return {"delta": self.delta, "delta_deg": math.degrees(self.delta),
                "eta": self.eta, "omega_sq": self.omega_sq,
                "A0_cw": self.A0[1], "A0_ccw": self.A0[-1], "C0": self.C0,
                "s_theta": self.s_theta, "s_t": self.s_t,
                "objective": self.objective, "n_spins": self.n_spins,
                "n_failed": self.n_failed,
                "n_by_direction": {str(k): v for k, v in self.n_by_direction.items()},
                "eta_at_bound": self.eta_at_bound}


def calibrate(obs: StrikeObservations, a: float, b: float,
              cfg: CalibrationConfig | None = None,
              deflector_spacing_rad: float = TWO_PI / 8) -> StrikeCalibration:
    """Fit ``(delta, eta, omega_sq)`` with ``A0`` and ``C0`` solved inside.

    Grid, then Nelder-Mead from the best few cells. The L1 surface has narrow
    basins that a purely local search started from one point does not reliably
    leave, which is why the grid comes first.

    Identifiability. ``omega_sq`` and the descent constants are correlated: a
    lower threshold sends the ball further and takes longer to get there, which
    ``A0`` and ``C0`` partly absorb. Fitting the threshold predicts better than
    holding it at the paper's value, which is also what stator does, so it is
    the default. Either way the fitted constants are effective, not physical.
    """
    cfg = cfg or CalibrationConfig()
    if len(obs) == 0:
        raise ValueError("calibrate(): no strike observations")

    s_theta = cfg.s_theta or deflector_spacing_rad / math.sqrt(12.0)
    s_t = cfg.s_t or S_T_INITIAL[cfg.annotation_mode]
    history = []

    d_del = float(cfg.delta_grid[1] - cfg.delta_grid[0]) if cfg.delta_grid.size > 1 else math.radians(15)
    d_eta = float(cfg.eta_grid[1] - cfg.eta_grid[0]) if cfg.eta_grid.size > 1 else 0.05
    d_wsq = float(cfg.omega_sq_grid[1] - cfg.omega_sq_grid[0]) if cfg.omega_sq_grid.size > 1 else 0.5
    scale = np.array([d_del, d_eta, d_wsq])
    w_floor = b * b + 0.25

    w_grid = cfg.omega_sq_grid if cfg.fit_omega_sq else np.array([cfg.omega_sq_fixed])
    n_free = 3 if cfg.fit_omega_sq else 2
    scale = scale[:n_free]

    def unpack(x, w_held):
        v = x * scale
        return float(v[0]), float(v[1]), (float(v[2]) if cfg.fit_omega_sq else w_held)

    def run(s_theta, s_t):
        grid = np.empty((cfg.delta_grid.size, cfg.eta_grid.size, w_grid.size))
        for i, delta in enumerate(cfg.delta_grid):
            for j, eta in enumerate(cfg.eta_grid):
                for k, wsq in enumerate(w_grid):
                    grid[i, j, k] = evaluate(obs, a, b, delta, eta, wsq, s_theta, s_t, cfg).S
        best = None
        for flat in np.argsort(grid, axis=None)[: max(1, cfg.n_starts)]:
            i, j, k = np.unravel_index(flat, grid.shape)
            w_held = float(w_grid[k])
            start = [cfg.delta_grid[i], cfg.eta_grid[j], w_held][:n_free]
            x0 = np.asarray(start, dtype=float) / scale

            def objective(x, w_held=w_held):
                delta, eta, omega_sq = unpack(x, w_held)
                if eta < 0.0 or eta > cfg.eta_max or omega_sq <= w_floor:
                    return 1e30
                return evaluate(obs, a, b, delta, eta, omega_sq, s_theta, s_t, cfg).S

            simplex = [x0] + [x0 + np.eye(n_free)[t] for t in range(n_free)]
            r = minimize(objective, x0, method="Nelder-Mead",
                         options={"initial_simplex": simplex, "xatol": 1e-3,
                                  "fatol": 1e-6, "maxiter": 800})
            if best is None or r.fun < best[0]:
                best = (float(r.fun), unpack(r.x, w_held))
        _, (delta, eta, omega_sq) = best
        delta = float(delta % TWO_PI)
        eta = float(max(0.0, eta))
        omega_sq = float(max(w_floor + 1e-9, omega_sq))
        ev = evaluate(obs, a, b, delta, eta, omega_sq, s_theta, s_t, cfg)
        return delta, eta, omega_sq, ev

    delta, eta, omega_sq, ev = run(s_theta, s_t)
    history.append({"pass": 1, "delta": delta, "eta": eta, "omega_sq": omega_sq,
                    "C0": ev.C0, "S": ev.S, "s_theta": s_theta, "s_t": s_t})
    if cfg.rescale:
        # Reweight by the residual scales the fit actually produced, clamped so
        # a poor first pass cannot silently switch a channel off.
        c = cfg.rescale_clamp
        fin_th = ev.r_theta[np.isfinite(ev.r_theta)]
        fin_t = ev.r_t[np.isfinite(ev.r_t)]
        if fin_th.size:
            s_theta = min(max(mad_scale(fin_th), S_THETA_FLOOR, s_theta / c), s_theta * c)
        if fin_t.size:
            s_t = min(max(mad_scale(fin_t), S_T_FLOOR, s_t / c), s_t * c)
        delta, eta, omega_sq, ev = run(s_theta, s_t)
        history.append({"pass": 2, "delta": delta, "eta": eta, "omega_sq": omega_sq,
                        "C0": ev.C0, "S": ev.S, "s_theta": s_theta, "s_t": s_t})

    n_by_dir = {sgn: int((obs.s == sgn).sum()) for sgn in (1, -1)}
    return StrikeCalibration(delta, eta, omega_sq, ev.A0, ev.C0, s_theta, s_t,
                             ev.S, len(obs), ev.n_failed, n_by_dir,
                             eta >= cfg.eta_max - 1e-3,
                             ev.r_theta, ev.r_t, history)


def profile(obs: StrikeObservations, a, b, cal: StrikeCalibration,
            cfg: CalibrationConfig | None = None, which: str = "delta", n: int = 180):
    """1-D objective profile through the optimum, for a basin sanity check."""
    cfg = cfg or CalibrationConfig()
    if which == "delta":
        xs = np.linspace(0.0, TWO_PI, n, endpoint=False)
        f = lambda x: evaluate(obs, a, b, x, cal.eta, cal.omega_sq, cal.s_theta, cal.s_t, cfg).S
    elif which == "eta":
        xs = np.linspace(0.0, cfg.eta_max, n)
        f = lambda x: evaluate(obs, a, b, cal.delta, x, cal.omega_sq, cal.s_theta, cal.s_t, cfg).S
    elif which == "omega_sq":
        xs = np.linspace(b * b + 0.3, 12.0, n)
        f = lambda x: evaluate(obs, a, b, cal.delta, cal.eta, x, cal.s_theta, cal.s_t, cfg).S
    else:
        raise ValueError(f"unknown profile coordinate {which!r}")
    return xs, np.array([f(x) for x in xs])
