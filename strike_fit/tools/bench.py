"""Timing and regression harness for the strike calibration.

Times one full fit and a leave-one-out over the first ``--folds`` strike spins,
and records or checks every fitted number and every held-out prediction. The
point is that speed work must not move results: record a golden file with the
code before a change, check against it after.

    python -m strike_fit.tools.bench --synth 120 --folds 4 --record out/golden_synth.json
    python -m strike_fit.tools.bench --data strike_fit/out/rc_208.jsonl --folds 8 \\
        --check strike_fit/out/golden_rc208.json

Run from the repo root, in a venv where ``stator`` is installed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

from strike_fit import StrikeRim, synth
from strike_fit.calibrate import CalibrationConfig
from strike_fit.crossval import FoldResult, leave_one_out, summary
from strike_fit.data import Dataset

PARAM_KEYS = ("a", "b", "delta", "eta", "omega_sq", "A0_cw", "A0_ccw", "C0", "lap_floor")
CAL_KEYS = ("objective", "s_theta", "s_t", "n_failed")


def load_rim(records, header, skip_id=None, M=4, Y=6):
    rim = StrikeRim(M=M, Y=Y, n_deflectors=header.n_deflectors,
                    config=CalibrationConfig(annotation_mode=header.annotation_mode))
    for r in records:
        if r["spin_id"] == skip_id:
            continue
        st = r["strike"]
        try:
            rim.add_timing(r["spin_id"], r["clicks_s"],
                           st["t_s"] if st else None, st["deflector"] if st else None,
                           1.0 if r["direction"] == "cw" else -1.0)
        except ValueError:
            pass
    return rim


def summarise(rim) -> dict:
    p = rim.params
    out = dict(zip(PARAM_KEYS, (p.ball_params.a, p.ball_params.b, p.dep_params.delta,
                                p.dep_params.eta, p.dep_params.omega_sq,
                                p.descent_params.A0_cw, p.descent_params.A0_ccw,
                                p.descent_params.C0, rim.lap_floor)))
    cal = rim.calibration
    out.update({k: getattr(cal, k) for k in CAL_KEYS})
    return out


def fold_record(r: FoldResult) -> dict:
    params = {k: r.fit[k] for k in PARAM_KEYS + CAL_KEYS}
    if r.prediction is not None:
        prediction = {"theta": r.prediction.theta, "t_f": r.prediction.t_f,
                      "deflector": r.prediction.deflector}
    elif r.error is not None and not r.error.startswith("needs "):
        prediction = {"error": r.error}
    else:
        prediction = None
    return {"id": r.id, "params": params, "prediction": prediction}


def run(ds: Dataset, n_folds: int, workers: int | None) -> dict:
    records = ds.records
    t = time.perf_counter()
    rim = load_rim(records, ds.header).fit()
    t_fit = time.perf_counter() - t

    # the spins add_timing accepts, as the batch arguments cross_validate takes
    spins = [r for r in records if load_rim([r], ds.header).n_timings]
    ids = [r["spin_id"] for r in spins]
    tss = [r["clicks_s"] for r in spins]
    t_strikes = [r["strike"]["t_s"] if r["strike"] else None for r in spins]
    deflectors = [r["strike"]["deflector"] if r["strike"] else None for r in spins]
    ss = [1.0 if r["direction"] == "cw" else -1.0 for r in spins]
    hold_out = [r["spin_id"] for r in spins if r["strike"] is not None][:n_folds]

    t = time.perf_counter()
    folds = leave_one_out(ids, tss, t_strikes, deflectors, ss, hold_out=hold_out,
                          n_deflectors=ds.header.n_deflectors, workers=workers,
                          config=CalibrationConfig(annotation_mode=ds.header.annotation_mode))
    t_loo = time.perf_counter() - t
    return {"full": summarise(rim), "folds": [fold_record(r) for r in folds],
            "summary": summary(folds),
            "timing": {"fit_s": t_fit, "loo_s": t_loo, "n_folds": len(hold_out),
                       "workers": workers, "n_spins": len(records),
                       "n_strikes": rim.calibration.n_spins}}


def _compare(path, got, ref, tol, worst):
    if isinstance(ref, dict):
        if not isinstance(got, dict) or set(got) != set(ref):
            worst.append((math.inf, path, got, ref))
            return
        for k in ref:
            _compare(f"{path}.{k}", got[k], ref[k], tol, worst)
    elif isinstance(ref, list):
        if not isinstance(got, list) or len(got) != len(ref):
            worst.append((math.inf, path, got, ref))
            return
        for i, (g, r) in enumerate(zip(got, ref)):
            _compare(f"{path}[{i}]", g, r, tol, worst)
    elif isinstance(ref, float):
        if not isinstance(got, (int, float)):
            worst.append((math.inf, path, got, ref))
        elif not (math.isnan(ref) and math.isnan(got)):
            d = abs(got - ref) / max(1.0, abs(ref))
            if not d <= tol:
                worst.append((d, path, got, ref))
    elif got != ref:
        worst.append((math.inf, path, got, ref))


def compare(got: dict, ref: dict, tol: float) -> list:
    worst = []
    _compare("", {k: got[k] for k in ("full", "folds")},
             {k: ref[k] for k in ("full", "folds")}, tol, worst)
    return sorted(worst, key=lambda w: -w[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="JSONL dataset")
    src.add_argument("--synth", type=int, help="generate this many synthetic spins")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=None,
                    help="fold processes (default: every core; 1 runs folds in this process)")
    ap.add_argument("--record", help="write results to this JSON file")
    ap.add_argument("--check", help="compare results against this JSON file")
    ap.add_argument("--tol", type=float, default=1e-9, help="relative tolerance for --check")
    args = ap.parse_args(argv)

    ds = Dataset.read_jsonl(args.data) if args.data else synth.generate(args.synth, eta=0.2, seed=args.seed)
    result = run(ds, args.folds, args.workers)
    tm, sm = result["timing"], result["summary"]
    print(f"spins {tm['n_spins']}, strikes {tm['n_strikes']}: fit {tm['fit_s']:.2f} s, "
          f"{tm['n_folds']} folds {tm['loo_s']:.2f} s "
          f"({tm['loo_s'] / max(1, tm['n_folds']):.2f} s/fold)")
    print(f"held out: {sm['n_predicted']}/{sm['n']} predicted, hit rate {sm['hit_rate']:.3f}, "
          f"median |dt| {sm['median_abs_dt']:.3f} s, errors {sm['n_errors']}")

    if args.record:
        with open(args.record, "w") as fh:
            json.dump(result, fh, indent=1)
        print(f"recorded {args.record}")
    if args.check:
        with open(args.check) as fh:
            ref = json.load(fh)
        worst = compare(result, ref, args.tol)
        if worst:
            print(f"MISMATCH: {len(worst)} values beyond tol {args.tol:g}")
            for d, path, g, r in worst[:15]:
                print(f"  {path}: got {g!r} ref {r!r} (rel {d:.3g})")
            return 1
        ref_tm = ref.get("timing", {})
        if ref_tm.get("fit_s"):
            print(f"match within {args.tol:g}; fit {ref_tm['fit_s'] / tm['fit_s']:.1f}x, "
                  f"folds {ref_tm['loo_s'] / max(tm['loo_s'], 1e-9):.1f}x faster than the recording")
    return 0


if __name__ == "__main__":
    sys.exit(main())
