"""Held-out evaluation of :class:`~strike_fit.rim.StrikeRim`, one process per fold.

Each fold refits the whole model -- backbone, lap floor, ``To`` and the strike
calibration -- on every spin not held out, then predicts each held-out spin
from its crossings ending ``M`` laps before its strike, exactly as live play
would. Folds are independent, so they run in a process pool with the compiled
kernels pinned to one thread per worker; a single worker instead runs in this
process and lets the calibration grid use every core.

Results do not depend on the worker count: the grid is the same arithmetic on
one thread or many.

The pool starts its workers with ``spawn``, which re-imports the calling
script, so a script that calls these functions needs the usual guard::

    if __name__ == "__main__":
        results = leave_one_out(ids, crossings, t_strikes, deflectors, senses)
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

from .calibrate import CalibrationConfig
from .rim import DEFAULT_M, DEFAULT_N_DEFLECTORS, DEFAULT_Y, StrikePrediction, StrikeRim


@dataclass(frozen=True)
class FoldResult:
    """One held-out spin.

    ``t_strike`` and ``t_strike_pred`` are on the spin's own clock, the clock
    its crossings were supplied on. ``fit`` holds the fold's fitted numbers as
    plain floats: ``a``, ``b``, ``lap_floor`` and ``StrikeCalibration.to_dict()``.
    ``error`` says why there is no prediction when the fit or the predict
    refused; a ball that never exits leaves ``prediction`` None with no error,
    as ``StrikeRim.predict`` does.
    """
    id: str
    s: float
    deflector: int | None
    t_strike: float | None
    prediction: StrikePrediction | None
    t_strike_pred: float | None
    fit: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def hit(self) -> bool:
        return self.prediction is not None and self.prediction.deflector == self.deflector


@dataclass(frozen=True)
class _Spins:
    ids: list
    timestamps: list
    t_strikes: list
    deflectors: list
    ss: list
    M: int
    Y: int
    n_deflectors: int
    config: CalibrationConfig | None


def _fit_summary(rim: StrikeRim) -> dict:
    p = rim.params
    out = {"a": p.ball_params.a, "b": p.ball_params.b, "lap_floor": rim.lap_floor}
    out.update(rim.calibration.to_dict())
    return out


def _run_fold(data: _Spins, held_out: list) -> list[FoldResult]:
    held = set(held_out)
    rim = StrikeRim(M=data.M, Y=data.Y, n_deflectors=data.n_deflectors, config=data.config)
    for j, id_ in enumerate(data.ids):
        if id_ not in held:
            rim.add_timing(id_, data.timestamps[j], data.t_strikes[j], data.deflectors[j], data.ss[j])

    fit, fit_error = {}, None
    try:
        rim.fit()
        fit = _fit_summary(rim)
    except RuntimeError as e:
        fit_error = str(e)

    index = {id_: j for j, id_ in enumerate(data.ids)}
    results = []
    for id_ in held_out:
        j = index[id_]
        ts = list(data.timestamps[j])
        s = float(data.ss[j])
        pred = t_pred = None
        error = fit_error
        if error is None:
            if len(ts) < data.M + data.Y + 1:
                error = f"needs {data.M + data.Y + 1} crossings, has {len(ts)}"
            else:
                anchor = len(ts) - 1 - data.M
                try:
                    pred = rim.predict(ts[:anchor + 1], s)
                except ValueError as e:
                    error = str(e)
                if pred is not None:
                    t_pred = float(ts[anchor]) + pred.t_f
        t_strike = data.t_strikes[j]
        deflector = data.deflectors[j]
        results.append(FoldResult(str(id_), s,
                                  None if deflector is None else int(deflector),
                                  None if t_strike is None else float(t_strike),
                                  pred, t_pred, fit, error))
    return results


# One copy of the spins per worker, sent once rather than with every fold.
_WORKER_SPINS: _Spins | None = None


def _init_worker(data: _Spins):
    global _WORKER_SPINS
    _WORKER_SPINS = data
    import numba
    numba.set_num_threads(1)


def _run_fold_in_worker(held_out: list) -> list[FoldResult]:
    return _run_fold(_WORKER_SPINS, held_out)


def cross_validate(ids, timestamps, t_strikes, deflectors, ss, folds, *,
                   M: int = DEFAULT_M, Y: int = DEFAULT_Y,
                   n_deflectors: int = DEFAULT_N_DEFLECTORS,
                   config: CalibrationConfig | None = None,
                   workers: int | None = None) -> list[FoldResult]:
    """Fit once per fold without its spins, predict those spins.

    The first five arguments are ``StrikeRim.add_timing``'s batch form.
    ``folds`` is a list of held-out id lists. Returns one :class:`FoldResult`
    per held-out id, in fold order. ``workers`` defaults to every core.
    """
    n = len(ids)
    if not (len(timestamps) == len(t_strikes) == len(deflectors) == len(ss) == n):
        raise ValueError("cross_validate(): argument lists must be the same length")
    if len(set(ids)) != n:
        raise ValueError("cross_validate(): spin ids must be unique")
    known = set(ids)
    folds = [list(f) for f in folds]
    for f in folds:
        missing = [i for i in f if i not in known]
        if missing:
            raise ValueError(f"cross_validate(): unknown held-out ids {missing[:5]}")

    data = _Spins(list(ids), [list(map(float, t)) for t in timestamps], list(t_strikes),
                  list(deflectors), [float(s) for s in ss], int(M), int(Y), int(n_deflectors),
                  config)
    workers = min(workers or os.cpu_count() or 1, len(folds))
    if workers <= 1:
        return [r for f in folds for r in _run_fold(data, f)]

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=_init_worker, initargs=(data,)) as pool:
        return [r for fold in pool.map(_run_fold_in_worker, folds) for r in fold]


def leave_one_out(ids, timestamps, t_strikes, deflectors, ss, *, hold_out=None,
                  **kwargs) -> list[FoldResult]:
    """One fold per spin in ``hold_out``.

    By default every spin that could calibrate is held out in turn: it carries
    a strike and enough crossings for the anchor to clear the ``To`` window.
    """
    if hold_out is None:
        M = kwargs.get("M", DEFAULT_M)
        Y = kwargs.get("Y", DEFAULT_Y)
        hold_out = [id_ for id_, ts, t in zip(ids, timestamps, t_strikes)
                    if t is not None and len(ts) >= M + Y + 1]
    return cross_validate(ids, timestamps, t_strikes, deflectors, ss,
                          [[id_] for id_ in hold_out], **kwargs)


def summary(results: list[FoldResult]) -> dict:
    """Hit rate and strike-time error over the results that carry a prediction."""
    scored = [r for r in results if r.prediction is not None and r.deflector is not None]
    dt = [abs(r.t_strike_pred - r.t_strike) for r in scored]
    return {"n": len(results), "n_predicted": len(scored),
            "hit_rate": sum(r.hit for r in scored) / len(scored) if scored else math.nan,
            "median_abs_dt": statistics.median(dt) if dt else math.nan,
            "n_errors": sum(r.error is not None for r in results)}
