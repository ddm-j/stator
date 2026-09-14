"""Convert ``roulette_timings`` hand measurements into the strike-fit JSONL contract.

Usage (from the repo root)::

    python -m strike_fit.tools.convert_measurements --video rc_208 \
        --timings ~/Projects/roulette_timings --out strike_fit/out/rc_208.jsonl

Per ball event (see ``DATA.md`` in the timings repo):

* ``clicks_s``: the ``ball_cross`` pass when present (every crossing up to the
  strike, the new convention), else ``ball_lap`` (on-rim laps only, the old
  convention); re-zeroed on the first crossing. Which one was used is recorded
  in ``flags`` (``CLICKS_BALL_LAP`` when it was the old convention).
* ``direction``: from the ``direction`` measure on the ball event (``1`` -> cw).
* ``strike``: ``{t_s, deflector}`` from the ``strike`` and ``deflector``
  measures, relative to the first crossing; ``null`` when either is absent.
* ``legacy``: ``angle_bin`` (1..36, 10-degree bins, lab frame, clockwise from
  the CS mark), ``departure_s`` (relative), ``departure_index`` -- stored raw
  for the baseline model; the candidate view never sees them.
* ``truth``: never present for real data.

Usability: a ball event is skipped when its own ``status`` is not ``active``,
when it carries an ``exclude``, or when its parent wheel spin carries a
``review`` with status ``bad``. Rows on both kinds of event keep their file
clock (seconds into the video); only differences are stored here.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from strike_fit.data import Dataset, Header  # noqa: E402

FLAG_CLICKS_BALL_LAP = "CLICKS_BALL_LAP"
FLAG_NO_DIRECTION = "NO_DIRECTION"


def load_rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def group(rows):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["event_id"]][r["measure"]].append(r)
    return by


def sequence(rows):
    """Indexed rows -> list of floats ordered by index."""
    return [float(r["value"]) for r in sorted(rows, key=lambda r: int(r["index"]))]


def usable(event_id, events):
    ev = events.get(event_id)
    if ev is None:
        return False, "unknown_event"
    if ev.get("status", "active") != "active":
        return False, f"status={ev.get('status')}"
    if ev.get("exclude"):
        return False, f"exclude={ev.get('exclude')}"
    parent = events.get(ev.get("wheel_spin") or "")
    if parent and isinstance(parent.get("review"), dict) and parent["review"].get("status") == "bad":
        return False, f"review={parent['review'].get('reason')}"
    return True, ""


def convert(timings_root, video, annotation_mode="reactive", fps=None, require_usable=True):
    data_dir = os.path.join(timings_root, "data")
    rows = load_rows(os.path.join(data_dir, "measurements.csv"))
    events = json.load(open(os.path.join(data_dir, "events.json")))
    videos = json.load(open(os.path.join(data_dir, "videos.json")))
    wheels = json.load(open(os.path.join(data_dir, "wheels.json")))

    vinfo = videos.get(video, {}) if isinstance(videos, dict) else {}
    wheel_id = vinfo.get("wheel") or vinfo.get("wheel_id")
    n_deflectors = (wheels.get(wheel_id, {}) or {}).get("n_deflectors") if wheel_id else None
    if n_deflectors is None:
        # fall back: the only wheel with a count
        counted = [w for w in wheels.values() if w.get("n_deflectors")]
        if len(counted) == 1:
            n_deflectors = counted[0]["n_deflectors"]
        else:
            raise SystemExit(f"cannot determine n_deflectors for video {video!r}; pass --n-deflectors")
    fps = fps or vinfo.get("fps")

    by = group(rows)
    records, skipped = [], collections.Counter()
    for event_id in sorted(by):
        vid, _, tag = event_id.partition(".")
        if vid != video or not tag.startswith("B"):
            continue
        m = by[event_id]
        if require_usable:
            ok, why = usable(event_id, events)
            if not ok:
                skipped[why] += 1
                continue
        flags = []
        if "ball_cross" in m:
            clicks = sequence(m["ball_cross"])
        elif "ball_lap" in m:
            clicks = sequence(m["ball_lap"])
            flags.append(FLAG_CLICKS_BALL_LAP)
        else:
            skipped["no_clicks"] += 1
            continue
        if len(clicks) < 2 or any(b <= a for a, b in zip(clicks, clicks[1:])):
            skipped["bad_clicks"] += 1
            continue
        t0 = clicks[0]
        if "direction" not in m:
            skipped["no_direction"] += 1
            continue
        s = int(float(m["direction"][0]["value"]))
        direction = "cw" if s > 0 else "ccw"

        strike = None
        if "strike" in m and "deflector" in m:
            t_s = float(m["strike"][0]["value"]) - t0
            d = int(float(m["deflector"][0]["value"]))
            if t_s > 0 and 0 <= d < n_deflectors:
                strike = {"t_s": t_s, "deflector": d}
            else:
                flags.append("BAD_STRIKE")

        legacy = {}
        if "angle" in m:
            legacy["angle_bin"] = int(float(m["angle"][0]["value"]))
        if "departure" in m:
            legacy["departure_s"] = float(m["departure"][0]["value"]) - t0
        if "departure_index" in m:
            legacy["departure_index"] = int(float(m["departure_index"][0]["value"]))
        if "pocket" in m:
            legacy["pocket"] = m["pocket"][0]["value"]

        records.append({
            "spin_id": event_id,
            "direction": direction,
            "clicks_s": [c - t0 for c in clicks],
            "strike": strike,
            "legacy": legacy,
            "rotor_clicks_s": None,
            "rotor_angles_rad": None,
            "flags": flags,
        })
    header = Header(n_deflectors=int(n_deflectors), annotation_mode=annotation_mode,
                    fps=fps, reference_deflector=0, video=video,
                    extra={"source": "roulette_timings/data/measurements.csv",
                           "skipped": dict(skipped)})
    return Dataset(header, records), skipped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--timings", default=os.path.expanduser("~/Projects/roulette_timings"))
    ap.add_argument("--video", default="rc_208")
    ap.add_argument("--out", default=None)
    ap.add_argument("--annotation-mode", default="reactive", choices=["reactive", "mechanical"])
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--all", action="store_true", help="include events the registry marks unusable")
    args = ap.parse_args(argv)
    ds, skipped = convert(args.timings, args.video, args.annotation_mode, args.fps, not args.all)
    out = args.out or os.path.join(ROOT, "strike_fit", "out", f"{args.video}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.write_jsonl(out)
    n_strike = sum(r["strike"] is not None for r in ds.records)
    n_cross = sum(FLAG_CLICKS_BALL_LAP not in r["flags"] for r in ds.records)
    n_legacy = sum("angle_bin" in r["legacy"] for r in ds.records)
    print(f"wrote {out}: {len(ds)} spins ({n_strike} with strike, {n_cross} on ball_cross, "
          f"{n_legacy} with legacy departure); skipped {dict(skipped)}")


if __name__ == "__main__":
    main()
