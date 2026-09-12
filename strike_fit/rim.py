"""``StrikeRim``: stator's ``Rim`` with the strike as its calibration target.

The two models share everything except what they are calibrated on:

===========================  ==========================  ==========================
stage                        stator ``Rim``              ``StrikeRim``
===========================  ==========================  ==========================
per-spin ``a``/``b``         ``stator.fit_ab``           same call
pooling                      median of ``a``, ``|b|``    same
lap floor                    ``stator.fit_lap_floor``    same call
ball state at the anchor     ``stator.estimate_To``      same call
calibration observable       departure angle             strike angle **and** time
calibration fit              ``stator.fit_departure``    :mod:`strike_fit.calibrate`
what ``predict`` returns     rim exit                    deflector strike
===========================  ==========================  ==========================

So a spin is reduced to the same ``To`` by the same estimator in both models,
and the exit equations are stator's own (the vectorised forms in
:mod:`strike_fit.model` are verified equal to ``predict_theta`` and
``predict_tf`` to machine precision). The difference is the calibration
channel, which is the whole point of the experiment.

Angles. The strike angle handed to the calibration is total travel from the
anchor crossing to the strike, revolutions included: ``2*pi`` per crossing after
the anchor, plus the struck deflector's angle within the final revolution.
Deflector angles run in ``(0, 2*pi]``, so a strike on the reference deflector
is a whole revolution of travel rather than none, matching the annotation
convention that tallies such a strike once, as a strike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import stator

from . import model
from .calibrate import CalibrationConfig, StrikeCalibration, StrikeObservations, calibrate
from .conventions import TWO_PI, deflector_spacing

DEFAULT_M = 4
DEFAULT_Y = 6
DEFAULT_N_DEFLECTORS = 8


# ----------------------------------------------------------------------------
# Deflector geometry
# ----------------------------------------------------------------------------

def deflector_travel_angle(d, s, n_deflectors: int = DEFAULT_N_DEFLECTORS) -> float:
    """Travel-frame angle of physical deflector ``d``, in ``(0, 2*pi]``.

    Deflectors are numbered clockwise from the reference, which is deflector 0.
    A clockwise ball reaches deflector ``d`` after ``d`` spacings of travel, an
    anticlockwise one after ``N - d``. The reference deflector itself sits at a
    full revolution, not at zero: reaching it means having gone round.
    """
    d = int(d)
    if not 0 <= d < n_deflectors:
        raise ValueError(f"deflector {d} outside 0..{n_deflectors - 1}")
    idx = d % n_deflectors if s > 0 else (n_deflectors - d) % n_deflectors
    return TWO_PI if idx == 0 else idx * deflector_spacing(n_deflectors)


def deflector_from_travel_angle(theta, s, n_deflectors: int = DEFAULT_N_DEFLECTORS):
    """Physical deflector nearest to travel-frame angle ``theta``. Vectorised."""
    spacing = deflector_spacing(n_deflectors)
    k = np.rint(np.asarray(theta, dtype=float) / spacing).astype(int) % n_deflectors
    out = np.where(np.asarray(s) > 0, k, (n_deflectors - k) % n_deflectors)
    return out if out.ndim else int(out)


def strike_deflector(theta_s, s, n_deflectors: int = DEFAULT_N_DEFLECTORS):
    """Which deflector a ball at travel angle ``theta_s`` strikes, and how many
    reference crossings it made getting there.

    The ball sweeps forward into the next deflector, so the struck one is the
    first deflector angle at or beyond its position. Returns
    ``(deflector, n_crossings_after_the_first)``, which is exactly what an
    annotator tallying every crossing up to the strike would write down, and is
    the inverse of :func:`deflector_travel_angle` composed with the revolution
    count. Angles are in ``(0, 2*pi]`` throughout, so a strike on the reference
    deflector closes a revolution rather than opening one.
    """
    spacing = deflector_spacing(n_deflectors)
    n_extra = int(math.ceil(theta_s / TWO_PI)) - 1
    psi = theta_s - TWO_PI * n_extra                     # in (0, 2*pi]
    idx = int(math.ceil(psi / spacing - 1e-12)) % n_deflectors
    d = idx if s > 0 else (n_deflectors - idx) % n_deflectors
    return d, n_extra


# ----------------------------------------------------------------------------
# Records and parameters, shaped like stator's
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class StrikeTiming:
    """One spin. Mirrors ``stator``'s ``BallTiming``, strike in place of theta."""
    id: str
    timestamps: np.ndarray
    tk: np.ndarray                  # re-zeroed on the first crossing
    s: float
    theta: float | None = None      # total travel, first crossing to the strike
    t_strike: float | None = None   # strike time, relative to the first crossing
    deflector: int | None = None

    @property
    def has_strike(self) -> bool:
        return self.theta is not None


@dataclass(frozen=True)
class DescentParams:
    """The two constants stator has no equivalent of."""
    A0_cw: float = 0.0
    A0_ccw: float = 0.0
    C0: float = 0.0

    def arc(self, s) -> float:
        return self.A0_cw if s > 0 else self.A0_ccw

    def __repr__(self):
        return (f"DescentParams(A0_cw={self.A0_cw:.6f}, A0_ccw={self.A0_ccw:.6f}, "
                f"C0={self.C0:.6f})")


@dataclass(frozen=True)
class StrikeFitParams:
    """``stator.FitParams`` plus the descent block.

    ``ball_params`` and ``dep_params`` are stator's own types, so this object
    can be handed straight to ``stator.predict_theta`` / ``predict_tf`` and read
    with the same attribute names as stator's ``params``.
    """
    ball_params: object
    dep_params: object
    descent_params: DescentParams

    @property
    def fit_params(self):
        """The rim-stage parameters alone, as a ``stator.FitParams``."""
        return stator.FitParams(self.ball_params, self.dep_params)

    def __repr__(self):
        return (f"StrikeFitParams(a={self.ball_params.a:.6f}, b={self.ball_params.b:.6f}, "
                f"delta={self.dep_params.delta:.6f}, eta={self.dep_params.eta:.6f}, "
                f"omega_sq={self.dep_params.omega_sq:.6f}, "
                f"A0_cw={self.descent_params.A0_cw:.6f}, "
                f"A0_ccw={self.descent_params.A0_ccw:.6f}, "
                f"C0={self.descent_params.C0:.6f})")


@dataclass(frozen=True)
class StrikePrediction:
    """Mirrors ``stator``'s ``BallPrediction``.

    ``theta`` and ``t_f`` carry the same meaning they do there, travel and
    elapsed time from the last supplied crossing, but to the **strike** rather
    than to rim exit. ``theta_rim`` and ``t_rim`` expose the rim-exit stage for
    diagnostics.
    """
    theta: float
    t_f: float
    deflector: int
    theta_rim: float
    t_rim: float

    def __repr__(self):
        return (f"StrikePrediction(theta={self.theta:.6f}, t_f={self.t_f:.6f}, "
                f"deflector={self.deflector})")


# ----------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------

class StrikeRim:
    """Strike-calibrated rim model with ``stator.Rim``'s interface.

    ``M`` lead laps sit between the anchor crossing and the strike, and the
    ``To`` estimator looks back ``Y`` laps from that anchor, exactly as in
    stator, so each stage sees only what it would see live.
    """

    def __init__(self, M: int = DEFAULT_M, Y: int = DEFAULT_Y,
                 n_deflectors: int = DEFAULT_N_DEFLECTORS,
                 config: CalibrationConfig | None = None):
        self.M = int(M)
        self.Y = int(Y)
        self.n_deflectors = int(n_deflectors)
        self.config = config or CalibrationConfig()
        self._data: list[StrikeTiming] = []
        self._params: StrikeFitParams | None = None
        self._lap_floor = 0.0
        self._a_slope = 0.0
        self._calibration: StrikeCalibration | None = None
        self._fitted_ids: list[str] = []

    # -- construction from known parameters --------------------------------
    @classmethod
    def from_params(cls, a, b, delta, eta, omega_sq, A0_cw=0.0, A0_ccw=0.0, C0=0.0,
                    lap_floor=0.0, a_slope=None, M=DEFAULT_M, Y=DEFAULT_Y,
                    n_deflectors=DEFAULT_N_DEFLECTORS) -> "StrikeRim":
        rim = cls(M=M, Y=Y, n_deflectors=n_deflectors)
        rim._params = StrikeFitParams(stator.BallParams(a, b),
                                      stator.DepartureParams(delta, eta, omega_sq),
                                      DescentParams(A0_cw, A0_ccw, C0))
        rim._lap_floor = float(lap_floor)
        rim._a_slope = float(a if a_slope is None else a_slope)
        return rim

    # -- data ---------------------------------------------------------------
    def add_timing(self, id, timestamps, t_strike=None, deflector=None, s=1.0):
        """Add one spin, or a batch when ``id`` is a list.

        ``timestamps`` are crossings of the reference deflector on the source
        clock, every crossing up to the strike. ``t_strike`` and ``deflector``
        are the strike, on the same clock; omit both to add a spin that feeds
        the deceleration fit and the lap floor but not the calibration, which
        is legitimate because lap crossings cost nothing extra to record.
        """
        if not isinstance(id, str):
            ids, tss = id, timestamps
            t_strikes = t_strike if t_strike is not None else [None] * len(ids)
            deflectors = deflector if deflector is not None else [None] * len(ids)
            ss = s if isinstance(s, (list, tuple, np.ndarray)) else [s] * len(ids)
            if not (len(ids) == len(tss) == len(t_strikes) == len(deflectors) == len(ss)):
                raise ValueError("StrikeRim.add_timing(): argument lists must be the same length")
            for j in range(len(ids)):
                self.add_timing(ids[j], tss[j], t_strikes[j], deflectors[j], ss[j])
            return

        if s not in (1.0, -1.0, 1, -1):
            raise ValueError(f"StrikeRim.add_timing(): timing ID {id} direction s must be +1 or -1, got {s}")
        ts = np.asarray(timestamps, dtype=float)
        if ts.size <= 2:
            raise ValueError(f"StrikeRim.add_timing(): timing ID {id} must have more than two timestamps.")
        if not np.all(np.diff(ts) > 0):
            raise ValueError(f"StrikeRim.add_timing(): timing ID {id} has non-monotonic timestamps.")
        tk = ts - ts[0]

        theta = t_rel = None
        d = None
        if (t_strike is None) != (deflector is None):
            raise ValueError(f"StrikeRim.add_timing(): timing ID {id} needs both a strike time and a deflector, or neither.")
        if t_strike is not None:
            d = int(deflector)
            if not 0 <= d < self.n_deflectors:
                raise ValueError(f"StrikeRim.add_timing(): timing ID {id} deflector {d} outside 0..{self.n_deflectors - 1}")
            if float(t_strike) <= ts[-1]:
                raise ValueError(f"StrikeRim.add_timing(): timing ID {id} strike must follow the last crossing.")
            theta = deflector_travel_angle(d, s, self.n_deflectors) + TWO_PI * (ts.size - 1)
            t_rel = float(t_strike) - ts[0]

        self._data.append(StrikeTiming(str(id), ts, tk, float(s), theta, t_rel, d))

    # -- fitting ------------------------------------------------------------
    def fit(self):
        """Fit the model. Same call sequence as ``Rim::fit`` up to calibration."""
        if not self._data:
            return

        a_s, b_s, tks, anchors, keep = [], [], [], [], []
        for spin in self._data:
            n = spin.tk.size
            if n < self.M + self.Y + 1:        # anchor must clear the To window
                continue
            anchor = n - 1 - self.M
            p = stator.fit_ab(spin.tk.tolist())          # pooled constant, every lap
            if p.a <= 0.0 or p.b == 0.0:                 # degenerate fit
                continue
            a_s.append(p.a)
            b_s.append(abs(p.b))                         # sign of b is not identified
            tks.append(spin.tk[:anchor + 1].tolist())
            anchors.append(anchor)
            keep.append(spin)
        if not a_s:
            raise RuntimeError("StrikeRim.fit(): no spin produced usable ball parameters")

        a = float(np.median(a_s))
        b = float(np.median(b_s))

        self._lap_floor = float(stator.fit_lap_floor(tks, self.Y))
        self._a_slope = a

        To, theta_s, t_s, senses, ids = [], [], [], [], []
        for j, spin in enumerate(keep):
            To_j = float(stator.estimate_To(tks[j], anchors[j], self._lap_floor, self._a_slope, self.Y))
            if not spin.has_strike:
                continue
            To.append(To_j)
            theta_s.append(spin.theta - TWO_PI * anchors[j])
            t_s.append(spin.t_strike - spin.tk[anchors[j]])
            senses.append(spin.s)
            ids.append(spin.id)
        if not To:
            raise RuntimeError("StrikeRim.fit(): no spin carries a strike to calibrate on")

        obs = StrikeObservations(np.asarray(To), np.asarray(theta_s), np.asarray(t_s),
                                 np.asarray(senses, dtype=int), ids)
        cal = calibrate(obs, a, b, self.config, deflector_spacing(self.n_deflectors))

        self._calibration = cal
        self._fitted_ids = ids
        self._params = StrikeFitParams(
            stator.BallParams(a, b),
            stator.DepartureParams(cal.delta, cal.eta, cal.omega_sq),
            DescentParams(cal.A0[1], cal.A0[-1], cal.C0))
        return self

    # -- prediction ---------------------------------------------------------
    def predict(self, timestamps, s=1.0):
        """Predict the strike from crossings ending ``M`` laps before it.

        The last supplied crossing is the anchor and ``To`` comes from the ``Y``
        laps behind it, matching how ``fit`` built its own. Returns ``None``
        when the ball never reaches the exit condition, as stator's does.
        Pass a list of timestamp lists for a batch.
        """
        if len(timestamps) and isinstance(timestamps[0], (list, tuple, np.ndarray)):
            ss = s if isinstance(s, (list, tuple, np.ndarray)) else [s] * len(timestamps)
            return [self.predict(t, ss[j]) for j, t in enumerate(timestamps)]

        if self._params is None:
            raise RuntimeError("StrikeRim.predict(): rim has not been fit")
        ts = np.asarray(timestamps, dtype=float)
        if not np.all(np.diff(ts) > 0):
            raise ValueError("StrikeRim.predict(): ball lap timestamps are non-monotonic.")
        tk = ts - ts[0]
        if tk.size < self.Y + 1:
            raise ValueError(f"StrikeRim.predict(): need {self.Y + 1} crossings for a "
                             f"{self.Y} lap window, got {tk.size}")

        tk_win = tk[-(self.Y + 1):].tolist()
        m = len(tk_win) - 1
        To = float(stator.estimate_To(tk_win, m, self._lap_floor, self._a_slope, self.Y))

        theta_rim = stator.predict_theta(To, float(s), self._params.fit_params)
        if theta_rim is None:
            return None
        t_rim = float(stator.predict_tf(To, theta_rim, float(s), self._params.fit_params))

        arc = self._params.descent_params.arc(s)
        if not math.isfinite(arc):
            raise ValueError(
                f"StrikeRim.predict(): no strike was calibrated for s={float(s):+.0f}, so this "
                f"model cannot predict that direction. Calibrated: {self.calibrated_senses}. "
                f"On a table where the dealer reverses every round, splitting spins by "
                f"alternating index selects a single direction -- stratify by sense instead.")
        theta = float(theta_rim) + arc
        t_f = t_rim + self._params.descent_params.C0
        d = int(deflector_from_travel_angle(theta, s, self.n_deflectors))
        return StrikePrediction(theta, t_f, d, float(theta_rim), t_rim)

    # -- accessors ----------------------------------------------------------
    @property
    def params(self) -> StrikeFitParams | None:
        return self._params

    @property
    def calibrated_senses(self) -> list:
        """Directions the strike calibration actually saw."""
        if self._params is None:
            return []
        d = self._params.descent_params
        return [s for s, v in ((1, d.A0_cw), (-1, d.A0_ccw)) if math.isfinite(v)]

    @property
    def calibration(self) -> StrikeCalibration | None:
        """Residuals and search diagnostics from the strike fit."""
        return self._calibration

    @property
    def lap_floor(self) -> float:
        return self._lap_floor

    @property
    def a_slope(self) -> float:
        return self._a_slope

    @property
    def n_timings(self) -> int:
        return len(self._data)

    @property
    def fitted_ids(self) -> list:
        """Spins that reached the calibration, in order."""
        return list(self._fitted_ids)

    def __repr__(self):
        if self._params is None:
            return f"StrikeRim(M={self.M}, Y={self.Y}, timings={len(self._data)}, unfitted)"
        p = self._params
        return (f"StrikeRim(a={p.ball_params.a:.6f}, b={p.ball_params.b:.6f}, "
                f"delta={p.dep_params.delta:.6f}, eta={p.dep_params.eta:.6f}, "
                f"omega_sq={p.dep_params.omega_sq:.6f}, "
                f"A0_cw={p.descent_params.A0_cw:.6f}, A0_ccw={p.descent_params.A0_ccw:.6f}, "
                f"C0={p.descent_params.C0:.6f}, lap_floor={self._lap_floor:.6f})")
