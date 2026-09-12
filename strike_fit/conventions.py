"""Constants, frames and helpers shared by every module.

Deflector geometry lives in :mod:`strike_fit.rim`, next to the model that
defines it.

Two frames:

* **lab frame** -- fixed to the bowl. Angles increase clockwise from the
  reference deflector (ID 0). Deflector IDs and the low-point angle ``delta``
  live here.
* **travel frame** -- fixed to the ball's direction of travel, ``theta = 0`` at
  the reference deflector, increasing in the direction the ball moves. The
  paper's equations, ``theta_f``, ``A0`` and every residual live here.

The one and only conversion between the two is the sign ``s`` (``+1`` for a
clockwise spin, ``-1`` for anticlockwise): a lab angle ``x`` is travel angle
``s * x``. Hence ``phi = s * delta``, exactly as ``stator/physics/departure.h``
has it. Everything else derives from that one line.
"""

from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi

# Exit threshold (rad/s)^2. The paper's value, used as a search centre and as
# the default when a model is built from parameters rather than fitted.
OMEGA_F_SQ = 7.62

# Default deflector count; the dataset header overrides it.
N_DEFLECTORS_DEFAULT = 8

DIRECTION_SIGN = {"cw": 1, "ccw": -1}
SIGN_DIRECTION = {1: "cw", -1: "ccw"}


def direction_sign(direction) -> int:
    """``"cw"`` / ``"ccw"`` (or ``+1`` / ``-1``) to the travel sign ``s``."""
    if isinstance(direction, str):
        return DIRECTION_SIGN[direction]
    s = int(direction)
    if s not in (1, -1):
        raise ValueError(f"direction sign must be +1 or -1, got {direction!r}")
    return s


def deflector_spacing(n_deflectors: int) -> float:
    """``Delta = 2*pi / N_d``."""
    return TWO_PI / n_deflectors


def kappa(a) -> float:
    """Exit-equation constant ``kappa = 1 + (4a^2 + 1)/2`` (``A`` in predict.h)."""
    return 1.0 + 0.5 * (4.0 * a * a + 1.0)


def wrap(x):
    """Wrap to ``(-pi, pi]``. Works on scalars and arrays."""
    x = np.asarray(x, dtype=float)
    w = x - TWO_PI * np.floor((x + math.pi) / TWO_PI)
    # floor-based wrap yields [-pi, pi); move the -pi endpoint to +pi
    w = np.where(w == -math.pi, math.pi, w)
    return w if w.ndim else float(w)


def wrap_positive(x):
    """Wrap to ``[0, 2*pi)``."""
    x = np.asarray(x, dtype=float)
    w = np.mod(x, TWO_PI)
    return w if w.ndim else float(w)


def mad_scale(r):
    """Robust sigma: ``1.4826 * median(|r - median(r)|)``."""
    r = np.asarray(r, dtype=float)
    return float(1.4826 * np.median(np.abs(r - np.median(r))))
