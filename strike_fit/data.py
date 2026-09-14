"""Data contract (brief section 9) and the two views of a dataset.

A dataset is a header object followed by one record per spin (JSONL on
disk). Two typed views are derived from the records:

* :class:`CandidateSpin` -- exactly what a live observer has: lap clicks,
  direction, strike click and deflector ID. Nothing else.
* :class:`BaselineSpin` -- the same plus the ``legacy`` block (departure
  annotations) for the baseline model and diagnostics.

Synthetic datasets also carry a ``truth`` block per spin. Neither ``legacy``
nor ``truth`` is reachable from a :class:`CandidateSpin`; the candidate fit
calls :func:`assert_candidate_clean` on its inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

from .conventions import N_DEFLECTORS_DEFAULT, SIGN_DIRECTION, direction_sign

ANNOTATION_MODES = ("mechanical", "reactive")
FORBIDDEN_IN_CANDIDATE = ("legacy", "truth")


@dataclass(frozen=True)
class Header:
    n_deflectors: int = N_DEFLECTORS_DEFAULT
    annotation_mode: str = "mechanical"
    fps: Optional[float] = None
    reference_deflector: int = 0
    video: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.annotation_mode not in ANNOTATION_MODES:
            raise ValueError(f"annotation_mode must be one of {ANNOTATION_MODES}, got {self.annotation_mode!r}")
        if self.n_deflectors < 2:
            raise ValueError("n_deflectors must be >= 2")

    def to_dict(self) -> dict:
        d = {"n_deflectors": self.n_deflectors, "annotation_mode": self.annotation_mode,
             "fps": self.fps, "reference_deflector": self.reference_deflector}
        if self.video is not None:
            d["video"] = self.video
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Header":
        known = {"n_deflectors", "annotation_mode", "fps", "reference_deflector", "video"}
        return cls(n_deflectors=int(d["n_deflectors"]),
                   annotation_mode=d["annotation_mode"],
                   fps=d.get("fps"),
                   reference_deflector=int(d.get("reference_deflector", 0)),
                   video=d.get("video"),
                   extra={k: v for k, v in d.items() if k not in known})


@dataclass(frozen=True)
class Strike:
    t_s: float          # seconds after the first lap click
    deflector: int      # physical ID, 0..N_d-1, clockwise from the reference


@dataclass(frozen=True)
class CandidateSpin:
    spin_id: str
    direction: str                      # "cw" | "ccw"
    clicks: np.ndarray                  # lap clicks, clicks[0] == 0, strictly increasing
    strike: Optional[Strike]
    rotor_clicks: Optional[np.ndarray] = None
    rotor_angles: Optional[np.ndarray] = None   # rotor angle at each rotor click, radians from the first
    flags: tuple = ()

    @property
    def s(self) -> int:
        return direction_sign(self.direction)

    @property
    def n_clicks(self) -> int:
        return int(self.clicks.size)


@dataclass(frozen=True)
class BaselineSpin(CandidateSpin):
    legacy: dict = field(default_factory=dict)


def _validate_record(rec: dict, n_deflectors: int) -> None:
    clicks = np.asarray(rec["clicks_s"], dtype=float)
    if clicks.ndim != 1 or clicks.size < 2:
        raise ValueError(f"{rec.get('spin_id')}: need at least two clicks")
    if clicks[0] != 0.0:
        raise ValueError(f"{rec.get('spin_id')}: clicks_s[0] must be 0, got {clicks[0]}")
    if not np.all(np.diff(clicks) > 0):
        raise ValueError(f"{rec.get('spin_id')}: clicks_s must be strictly increasing")
    if rec["direction"] not in SIGN_DIRECTION.values():
        raise ValueError(f"{rec.get('spin_id')}: direction must be 'cw' or 'ccw'")
    st = rec.get("strike")
    if st is not None:
        d = int(st["deflector"])
        if not 0 <= d < n_deflectors:
            raise ValueError(f"{rec.get('spin_id')}: deflector {d} outside 0..{n_deflectors - 1}")
        if float(st["t_s"]) <= 0.0:
            raise ValueError(f"{rec.get('spin_id')}: strike must come after the first click")
    rc, ra = rec.get("rotor_clicks_s"), rec.get("rotor_angles_rad")
    if (rc is None) != (ra is None):
        # never guess a rotor angle per click: that is how full revolutions get assumed
        raise ValueError(f"{rec.get('spin_id')}: rotor_clicks_s and rotor_angles_rad must be given together")
    if rc is not None:
        rc, ra = np.asarray(rc, dtype=float), np.asarray(ra, dtype=float)
        if rc.ndim != 1 or rc.shape != ra.shape:
            raise ValueError(f"{rec.get('spin_id')}: rotor_angles_rad must have one angle per rotor click")
        if not (np.all(np.diff(rc) > 0) and np.all(np.diff(ra) > 0)):
            raise ValueError(f"{rec.get('spin_id')}: rotor clicks and angles must be strictly increasing")


def _spin_common(rec: dict) -> dict:
    st = rec.get("strike")
    rc = rec.get("rotor_clicks_s")
    ra = rec.get("rotor_angles_rad")
    return dict(
        spin_id=str(rec["spin_id"]),
        direction=rec["direction"],
        clicks=np.asarray(rec["clicks_s"], dtype=float),
        strike=None if st is None else Strike(float(st["t_s"]), int(st["deflector"])),
        rotor_clicks=None if rc is None else np.asarray(rc, dtype=float),
        rotor_angles=None if ra is None else np.asarray(ra, dtype=float),
        flags=tuple(rec.get("flags", ())),
    )


class Dataset:
    """Header plus raw spin records; views are built on demand."""

    def __init__(self, header: Header, records: Iterable[dict]):
        self.header = header
        self.records = list(records)
        for rec in self.records:
            _validate_record(rec, header.n_deflectors)

    def __len__(self) -> int:
        return len(self.records)

    # -- views ------------------------------------------------------------
    def candidate_view(self) -> list[CandidateSpin]:
        return [CandidateSpin(**_spin_common(rec)) for rec in self.records]

    def baseline_view(self) -> list[BaselineSpin]:
        return [BaselineSpin(**_spin_common(rec), legacy=dict(rec.get("legacy") or {}))
                for rec in self.records]

    def truth(self) -> list[Optional[dict]]:
        """Synthetic ground truth per spin (None for real data). Diagnostics only."""
        return [rec.get("truth") for rec in self.records]

    # -- io ----------------------------------------------------------------
    def write_jsonl(self, path) -> None:
        with open(path, "w") as fh:
            fh.write(json.dumps(self.header.to_dict()) + "\n")
            for rec in self.records:
                fh.write(json.dumps(_jsonable(rec)) + "\n")

    @classmethod
    def read_jsonl(cls, path) -> "Dataset":
        with open(path) as fh:
            lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        if not lines:
            raise ValueError(f"{path}: empty dataset")
        header = Header.from_dict(json.loads(lines[0]))
        return cls(header, (json.loads(ln) for ln in lines[1:]))


def _jsonable(obj: Any):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def assert_candidate_clean(spins: Iterable[CandidateSpin]) -> None:
    """Refuse anything that carries legacy or truth information (P1)."""
    for sp in spins:
        if not isinstance(sp, CandidateSpin):
            raise TypeError(f"candidate fit received {type(sp).__name__}, expected CandidateSpin")
        for name in FORBIDDEN_IN_CANDIDATE:
            if hasattr(sp, name):
                raise AssertionError(f"candidate fit received a spin exposing {name!r} ({sp.spin_id})")
