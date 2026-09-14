"""``Predictor``: stator's predictor with a :class:`~strike_fit.rim.StrikeRim`.

The rotor stage and the pocket mapping are stator's, imported rather than
reimplemented, so nothing about the wheel can drift between the two models.
The combination step is stator's arithmetic verbatim: signed ball travel minus
signed rotor travel, offset by half a pocket, resolved through the pocket
table. Ball and rotor timings share one physical reference, and the rotor's
reference is the zero pocket, which is what makes that subtraction meaningful.

The one difference is what the ball's travel and time refer to. stator drops
the ball at rim exit; this drops it at the deflector strike. The rotor is
therefore advanced to the strike, not to the departure, which is a difference
worth more than a pocket at a nominal rotor speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import stator

from .rim import StrikeRim


@dataclass(frozen=True)
class Prediction:
    """Mirrors ``stator``'s ``Prediction``, field for field.

    ``departure_time`` keeps stator's name so downstream code is identical, but
    in this model it is the absolute time of the **strike**. ``strike_time`` is
    the same number under the name that describes it.
    """
    departure_time: float       # absolute clock, ball timestamps' clock
    ball_travel: float          # radians, unsigned, arc-length CS
    wheel_travel: float         # radians, unsigned
    wheel_angle: float          # ball position in the wheel frame at the drop
    pocket: str
    deflector: int = -1         # struck deflector, this model's own output

    @property
    def strike_time(self) -> float:
        return self.departure_time

    def __repr__(self):
        return (f"Prediction(pocket={self.pocket}, deflector={self.deflector}, "
                f"strike_time={self.departure_time:.6f}, "
                f"ball_travel={self.ball_travel:.6f}, "
                f"wheel_travel={self.wheel_travel:.6f}, "
                f"wheel_angle={self.wheel_angle:.6f})")


class Predictor:
    """Ball plus rotor to a pocket. Same construction and call as stator's."""

    def __init__(self, rim: StrikeRim, wheel):
        self.rim = rim
        self.wheel = wheel

    def predict(self, ball_ts, wheel_ts, wheel_angles, ball_sense, wheel_sense):
        """Predict the pocket under the ball when it reaches the deflector.

        ``ball_ts`` ends ``M`` laps before the strike. ``wheel_ts`` is the two
        rotor presses that time the rotor and ``wheel_angles`` the rotor angle
        at each, radians from the pass's first press; any spacing is valid.
        Returns ``None`` when the ball stage cannot produce an exit, as
        stator's does.
        """
        ball_pred = self.rim.predict(ball_ts, ball_sense)
        if ball_pred is None:
            return None

        ball_travel = ball_pred.theta
        t_drop = ball_pred.t_f + float(ball_ts[-1])

        wheel_travel = float(self.wheel.predict(list(wheel_ts), list(wheel_angles), t_drop))

        W = (math.copysign(ball_travel, ball_sense)
             - math.copysign(wheel_travel, wheel_sense)
             + self.wheel.pkt_ang / 2.0)
        pocket = self.wheel.get_pkt_from_angle(W)

        return Prediction(t_drop, ball_travel, wheel_travel, W, pocket,
                          ball_pred.deflector)

    def __repr__(self):
        return "Predictor()"
