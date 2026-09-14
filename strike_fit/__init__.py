"""A strike-calibrated sibling to stator's rim model.

stator observes where the ball leaves the rim and predicts that. This model
observes where and when the ball hits a deflector and predicts that instead,
from data an observer can tap live: crossings of one reference deflector, the
spin sense, the strike click and which deflector was struck.

Everything the two models genuinely share is stator's own code, imported rather
than copied: the per-spin deceleration fit and its pooling, the lap floor, the
``To`` estimator, the exit equations, the rotor fit and the pocket table. Only
the calibration and the descent stage are new here.

The interface mirrors stator's so an experiment can swap one for the other::

    rim = StrikeRim(M=4, Y=6)
    rim.add_timing(spin_id, crossings, t_strike, deflector, s)
    rim.fit()
    pred = Predictor(rim, wheel).predict(ball_ts, wheel_ts, wheel_angles, ball_sense, wheel_sense)
"""

from stator import Wheel  # the rotor stage and pocket table, unchanged

from .calibrate import (CalibrationConfig, StrikeCalibration, StrikeObservations,
                        calibrate, profile)
from .crossval import FoldResult, cross_validate, leave_one_out
from .predictor import Prediction, Predictor
from .rim import (DescentParams, StrikeFitParams, StrikePrediction, StrikeRim,
                  StrikeTiming, deflector_from_travel_angle, deflector_travel_angle,
                  strike_deflector)

__all__ = [
    "Wheel", "StrikeRim", "Predictor", "Prediction", "StrikePrediction",
    "StrikeFitParams", "DescentParams", "StrikeTiming",
    "CalibrationConfig", "StrikeCalibration", "StrikeObservations",
    "calibrate", "profile",
    "cross_validate", "leave_one_out", "FoldResult",
    "deflector_travel_angle", "deflector_from_travel_angle", "strike_deflector",
]
