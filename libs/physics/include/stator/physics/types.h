#pragma once

#include <string>
#include <algorithm>

#include <stator/core/types.h>
#include <stator/core/constants.h>

using namespace stator::core;

namespace stator::physics
{

struct BallTiming
{
    const std::string id;
    const std::vector<real> timestamps;
    const std::vector<real> tk;
    const real theta;
    const real s;               // +1 clockwise, -1 anticlockwise

    BallTiming(const std::string_view id, const std::vector<real>& ts, real theta, real s = 1.0)
        : id { id }
        , timestamps { ts }
        , tk { timestamps_to_tk(ts) }
        , theta { theta + 2*pi*static_cast<real>(ts.size() - 1) }
        , s { s }
    {}

private:
    static std::vector<real> timestamps_to_tk(const std::vector<real>& timestamps)
    {
        std::vector<real> tk(timestamps.size(), 0.0);
        std::transform(timestamps.begin(), timestamps.end(), tk.begin(),
                        [&timestamps](real stamp){ return stamp - timestamps[0]; });
        return tk;
    }
};

struct BallParams
{
    // Parameters that Describe Ball Motion
    real a {};
    real b {};
};

struct DepartureParams
{
    real delta {};              // lab frame angle, reference mark to low point

    real eta {};
    real omega_sq {};
};

struct FitParams
{
    BallParams ball_params {};
    DepartureParams dep_params {};
};

struct BallPrediction
{
    real theta {};
    real t_f {};
};

struct WheelTiming
{
    const std::string id;
    const std::vector<real> timestamps;
    const std::vector<real> angles;     // rotor angle at each timestamp, radians

    WheelTiming(const std::string_view id, const std::vector<real>& ts, const std::vector<real>& angles)
        : id { id }
        , timestamps { ts }
        , angles { angles }
    {}
};

struct Prediction
{
    real departure_time {}; // Absolute time (not relative) of ball departure from rim
    real ball_travel {};     // Ball angular travel distance (radians, unsigned, arc-length CS)
    real wheel_travel {};    // Wheel angle at departure, from the timing's zero angle (radians, unsigned)
    real wheel_angle {};    // Ball location in wheel frame at departure
    std::string_view pocket {}; // Pocket corresponding to wheel angle
};

}
