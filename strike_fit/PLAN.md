# strike_fit

A sibling to stator's rim model, for A/B comparison. stator predicts where and
when the ball leaves the rim; this predicts where and when it hits a deflector.
That is the only intended difference between them.

Experiments live in `roulette-timings`, not here. This folder is a library.

## What it consumes

Per spin, all of it tappable live:

1. crossing times of one reference deflector, every crossing up to the strike
2. the spin sense
3. the strike time, on the same clock
4. which deflector was struck, an integer counted clockwise from the reference

Departure angles are never read. Spins with crossings but no strike are
accepted and feed the deceleration fit only.

## Interface

Mirrors stator's, so swapping models in an experiment is a one-line change.

```python
rim = StrikeRim(M=4, Y=6, n_deflectors=8)
rim.add_timing(spin_id, crossings, t_strike, deflector, s)   # t_strike/deflector may be None
rim.fit()
rim.predict(crossings, s)          # -> StrikePrediction(theta, t_f, deflector, ...)

Predictor(rim, Wheel.European()).predict(ball_ts, wheel_ts, ball_sense, wheel_sense)
```

`predict` takes the same arguments as stator's and returns the same field
names. `theta` and `t_f` are travel and elapsed time from the last supplied
crossing, but to the strike rather than to rim exit. `Prediction` keeps
stator's field names too, so `departure_time` carries the strike time;
`strike_time` is the same number under a truthful name. `params.ball_params`
and `params.dep_params` are stator's own types, and `params.fit_params` is a
real `stator.FitParams`.

## What is shared with stator, and verified so

Imported, not reimplemented: the per-spin `fit_ab` and its median pooling, the
lap floor, the `To` estimator, the exit equations, the rotor fit and the pocket
table. `strike_fit.Wheel` **is** `stator.Wheel`.

`tests/test_equivalence.py` pins the rest:

- identical crossings give identical `a`, `b`, lap floor and slope as `stator.Rim`
- with the descent constants zeroed, `predict` reproduces `stator.Rim.predict` exactly
- with the descent zeroed, the `Predictor` reproduces stator's pocket, wheel angle and rotor travel exactly
- the vectorised exit solver matches `predict_theta`/`predict_tf` to 1e-14 across both senses

## What is not shared

The calibration. stator fits `(delta, eta, omega_sq)` from departure angles
with an L1 line fit inside a search over `delta`. Here the observable is a
strike, which is a departure plus a descent, contributing an arc `A0` and a
time `C0`. Those sit inside the nonlinearity, so the search runs over
`(delta, eta, omega_sq)` with `A0` and `C0` solved exactly inside each
evaluation as medians. Same outer dimension as stator's.

The strike angle is unwrapped: total travel from the anchor, `2*pi` per
crossing after it plus the struck deflector's angle. Deflector angles run in
`(0, 2*pi]`, matching the annotation convention where a strike on the reference
deflector closes a revolution.

## Things learned that affect how results are read

- **The descent constants are effective, not physical.** They also absorb the
  `To` estimator's systematic error, which the exit angle amplifies by roughly
  ten radians per rad/s. `A0` is bounded to `(0, 2*pi)`: positive because the
  ball leaves the rim before it strikes, under a revolution because every
  crossing up to the strike is tapped. Removing the lower bound lets the fit
  place rim exit *after* the strike, which fits in sample and fails out of it.
- **`omega_sq` and the descent constants trade against each other.** Fitting
  `omega_sq` predicts better than holding it at the paper's 7.62 and matches
  what stator does, so it is the default. `cfg.fit_omega_sq = False` holds it.
- **Do not read `delta` across models.** The objective has a mirror basin near
  `delta + pi` with the offsets shifted to match. Compare predictions.
- **Split by sense, not by index.** On rc_208 the dealer reverses every round,
  so alternating spins selects one direction. A direction with no strikes now
  raises on `predict` instead of inventing an offset.

## Status on rc_208 (2026-09-10, annotation ongoing)

Sense-stratified half/half, held out, 57 predicted spins. Numbers move as more
strikes are annotated; treat as a smoke reading, not a result.

| | value |
|---|---|
| deflector hit rate | 0.79 - 0.86 depending on the tilt ceiling |
| always-dominant-diamond null | 0.86 |
| strike angle residual, median | +0.06 rad |
| strike angle residual, p10 to p90 | 1.6 rad, about two deflector spacings |
| strike time, median absolute error | 0.22 s |

Two caveats worth carrying into the comparison. The wheel is strongly
dominant-diamond, 87% of strikes on deflector 6 in **both** senses, so the
deflector channel is nearly saturated and the trivial null is hard to beat;
the time channel discriminates better. And the tilt runs to whatever ceiling
it is given, trading against `omega_sq` along a near-flat ridge, so it is not
identified on this data. `calibration.eta_at_bound` reports when that happens.

## Speed

The calibration objective is evaluated about ten thousand times per fit. Its
search runs on numba kernels in `_kernels.py`, which repeat `model.exit_angle`,
`model.fall_time` and `calibrate.evaluate` per spin; `tests/test_kernels.py`
pins them to the numpy forms, which remain the reference and still produce the
final diagnostics. The one algorithmic difference is the root polish: the
kernel uses safeguarded Newton (as `core/rtsafe.h`) where `model.exit_angle`
bisects, and the roots agree to about 1e-13 rad. The grid's forward model is
solved once per fit (in parallel) and only rescored by the rescaled second
pass.

Held-out evaluation lives in `crossval.py`. `leave_one_out` and
`cross_validate` refit per fold and predict the held-out spins live, one
process per fold; results are identical for any worker count. Scripts calling
them need an `if __name__ == "__main__":` guard, because workers are spawned.

    results = leave_one_out(ids, crossings, t_strikes, deflectors, senses)
    summary(results)        # hit rate, median |dt|, errors

| | before | now |
|---|---|---|
| one fit, rc_208 (55 strikes) | 11.3 s | 0.2 s |
| leave-one-out, rc_208, 55 folds | ~11 min | 2.3 s |
| leave-one-out, synthetic 200 spins, 191 folds | ~1 h | 12 s |

Nelder-Mead on the L1 surface amplifies last-digit rounding: perturbing `To`
by 1e-14 moves the rescaled `s_t` of some leave-one-out folds in the fourth
digit, with the old code as with the new. Compare predictions, not a fold's
scales, when checking a change against a recording.

`tools/bench.py` times a fit and a leave-one-out and records or checks every
fitted number and held-out prediction:

    python -m strike_fit.tools.bench --data strike_fit/out/rc_208.jsonl --folds 8 --record out/golden.json
    python -m strike_fit.tools.bench --data strike_fit/out/rc_208.jsonl --folds 8 --check out/golden.json

## Environment

Requires `stator` and `numba`, so it runs in the `roulette-timings` venv where
stator is installed. Tests: `python -m pytest strike_fit/tests` from the repo
root. The first import compiles the kernels and caches them in `__pycache__`.

`tools/convert_measurements.py` and `data.py` produce JSONL, used as a test
fixture only. The API consumes raw arrays; JSONL is not a pipeline stage.
