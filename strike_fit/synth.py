"""Synthetic data generator (brief section 7) in the delta framework.

Truth: ``(a, b)``, ``omega_f_sq_true``, tilt ``eta``, lab-frame low-point angle
``delta`` (travel phase ``phi = s * delta``), per-direction descent arcs
``A0[s]`` (travel frame, added to ``theta_f``), descent time ``C0``. The
descent arcs are kept separate from ``delta`` so the tests prove *prediction*
equivalence, not parameter equality (P9).

Per spin: level-equivalent speed ``omega0``, then ``theta_f, t_f`` from the
true model, strike angle ``theta_s = theta_f + A0[s] + N(0, sigma_A0^2)`` and
strike time ``t_f + C0 + bias(mode) + noise(mode)``. The struck deflector is
the first one at or beyond ``theta_s``, which is how a descending ball picks
one.

Crossings follow the annotation convention this model is built for: **every**
crossing of the reference up to the strike, on the rim or off it. A ball whose
descent carries it past the reference therefore contributes one more crossing,
and that crossing arrives early, because a ball that has left the rim cuts
inward and gets there sooner than the rim law says. Crossing times carry iid
Gaussian noise and are re-zeroed on the noisy first one, exactly as real data is.

Injected, flagged in ``truth``: hover (a late lobe at ``+T_rev`` with an extra
revolution of travel) and a missing strike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

from . import model
from .conventions import OMEGA_F_SQ, TWO_PI, wrap_positive
from .data import Dataset, Header
from .rim import strike_deflector

# Annotation-mode error model (brief section 6)
MODE_BIAS = {"mechanical": 0.0, "reactive": 0.20}
MODE_SIGMA = {"mechanical": 0.025, "reactive": 0.05}


@dataclass
class Truth:
    a: float = 0.0225
    b: float = math.sqrt(10.0 / 3.0)
    omega_f_sq: float = OMEGA_F_SQ
    eta: float = 0.2
    delta: float = 1.0                       # lab-frame low point angle (phi = s*delta)
    A0: dict = field(default_factory=lambda: {1: 0.5, -1: 0.7})
    C0: float = 0.45
    sigma_A0: float = 0.10                    # descent-arc scatter, rad; makes the oracle imperfect
    sigma_click: float = 0.03
    p_hover: float = 0.08
    hover_sigma: float = 0.15                 # relative jitter on the +T_rev delay
    p_missing: float = 0.05
    early_arrival: tuple = (0.05, 0.25)       # how early an off-rim crossing lands, s
    omega0_range: tuple = (10.0, 17.0)        # gives 12-15 crossings, as rc_208 spins have
    n_deflectors: int = 8
    legacy_bins: int = 36
    legacy_sigma: float = 0.008               # frame-read departure time, 60 fps

    @property
    def T_rev(self) -> float:
        return TWO_PI / math.sqrt(self.omega_f_sq)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["A0"] = {str(k): v for k, v in self.A0.items()}
        return d


def generate(n: int = 1000, eta: float = 0.2, seed: int = 0, mode: str = "mechanical",
             truth: Truth | None = None, **overrides) -> Dataset:
    """Build a synthetic :class:`Dataset` with ``n`` spins."""
    if truth is None:
        truth = Truth(eta=eta, **overrides)
    rng = np.random.default_rng(seed)
    a, b = truth.a, truth.b
    s = np.where(rng.random(n) < 0.5, 1, -1)
    omega0 = rng.uniform(*truth.omega0_range, n)
    phi = s * truth.delta
    pred = model.predict(a, b, omega0, truth.eta, phi, truth.omega_f_sq)
    # Truth must be exit-able; resample the rare flagged spin
    bad = pred.flags != 0
    while bad.any():
        omega0[bad] = rng.uniform(*truth.omega0_range, bad.sum())
        p2 = model.predict(a, b, omega0[bad], truth.eta, phi[bad], truth.omega_f_sq)
        pred.theta_f[bad], pred.t_f[bad], pred.flags[bad] = p2.theta_f, p2.t_f, p2.flags
        bad = pred.flags != 0

    hover = rng.random(n) < truth.p_hover
    missing = rng.random(n) < truth.p_missing
    bias = MODE_BIAS[mode]
    sig_s = MODE_SIGMA[mode]

    # Per-spin scalars are drawn up front so a spin's strike does not depend on
    # how many random numbers earlier spins happened to consume. That keeps two
    # runs that differ only in an injection rate comparable spin by spin.
    eps_arc = rng.normal(0.0, truth.sigma_A0, n)
    eps_time = rng.normal(0.0, sig_s, n)
    eps_hover = rng.normal(0.0, truth.hover_sigma, n)

    records = []
    for i in range(n):
        th_f, t_f = float(pred.theta_f[i]), float(pred.t_f[i])
        sign = int(s[i])

        # strike: rim exit plus the descent
        A0 = truth.A0[sign]
        theta_s = th_f + A0 + eps_arc[i]
        t_s = t_f + truth.C0 + bias + eps_time[i]
        if hover[i]:
            t_s += truth.T_rev * (1.0 + eps_hover[i])
            theta_s += TWO_PI
        deflector, n_extra = strike_deflector(theta_s, sign, truth.n_deflectors)

        # every crossing up to the strike, the ones past rim exit arriving early
        k = np.arange(n_extra + 1)
        t_clean = model.lap_times(k, a, b, omega0[i])
        off_rim = (TWO_PI * k > th_f) & (not hover[i])
        if off_rim.any():
            t_clean = t_clean - np.where(off_rim, rng.uniform(*truth.early_arrival, k.size), 0.0)
        t_noisy = t_clean + rng.normal(0.0, truth.sigma_click, k.size)
        t_noisy = np.maximum.accumulate(t_noisy)          # keep crossings ordered
        t0 = t_noisy[0]
        clicks = t_noisy - t0
        clicks[0] = 0.0
        t_s_rel = max(float(t_s - t0), float(clicks[-1]) + 1e-3)

        strike = None if missing[i] else {"t_s": t_s_rel, "deflector": int(deflector)}

        # what stator would be given for the same spin, for reference only
        lab_angle = wrap_positive(sign * th_f)
        legacy = {
            "angle_bin": int(math.floor(lab_angle / (TWO_PI / truth.legacy_bins))) + 1,
            "departure_s": float(t_f + rng.normal(0.0, truth.legacy_sigma) - t0),
            "departure_index": int(math.floor(th_f / TWO_PI)),
        }

        records.append({
            "spin_id": f"syn_{i:05d}",
            "direction": "cw" if sign > 0 else "ccw",
            "clicks_s": clicks.tolist(),
            "strike": strike,
            "legacy": legacy,
            "rotor_clicks_s": None,
            "rotor_angles_rad": None,
            "flags": [],
            "truth": {
                "omega0": float(omega0[i]),
                "theta_f": th_f,
                "t_f": t_f,
                "theta_s": float(theta_s),
                "t0_offset": float(t0),
                "hover": bool(hover[i]),
                "missing_strike": bool(missing[i]),
                "n_crossings": int(n_extra + 1),
                "n_off_rim": int(off_rim.sum()),
            },
        })

    header = Header(n_deflectors=truth.n_deflectors, annotation_mode=mode, fps=60.0,
                    reference_deflector=0, video="synthetic",
                    extra={"truth": truth.to_dict(), "seed": seed, "bias": bias})
    return Dataset(header, records)


def oracle_predictions(ds: Dataset, truth: Truth | None = None):
    """Deflector and strike-time predictions from the *true* parameters.

    Uses the true ``omega0`` (no fitted backbone) and the true
    ``(eta, delta, A0, C0)`` with the generator's own angle-to-deflector map.
    It is not a perfect predictor: the descent arc scatters by ``sigma_A0``,
    which is the irreducible part of the deflector channel.
    """
    if truth is None:
        truth = truth_from_header(ds.header)
    s = np.array([1 if r["direction"] == "cw" else -1 for r in ds.records])
    om0 = np.array([r["truth"]["omega0"] for r in ds.records])
    t0 = np.array([r["truth"]["t0_offset"] for r in ds.records])
    pred = model.predict(truth.a, truth.b, om0, truth.eta, s * truth.delta, truth.omega_f_sq)
    d_pred = np.full(len(ds), -1, dtype=int)
    for i, (th, sgn) in enumerate(zip(pred.theta_f, s)):
        if np.isfinite(th):
            d_pred[i] = strike_deflector(float(th) + truth.A0[int(sgn)], int(sgn),
                                         truth.n_deflectors)[0]
    bias = ds.header.extra.get("bias", 0.0)
    t_pred = pred.t_f + truth.C0 + bias - t0
    return d_pred, t_pred


def truth_from_header(header: Header) -> Truth:
    d = dict(header.extra["truth"])
    d["A0"] = {int(k): v for k, v in d["A0"].items()}
    d["omega0_range"] = tuple(d["omega0_range"])
    return Truth(**d)
