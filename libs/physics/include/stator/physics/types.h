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

    BallTiming(const std::string_view id, const std::vector<real>& ts, real theta)
        : id { id }
        , timestamps { ts }
        , tk { timestamps_to_tk(ts) }
        , theta { theta + 2*pi*static_cast<real>(ts.size() - 1) }
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
    real phi {};
    real eta {};
    real omega_sq {};
};

struct FitParams
{
    BallParams ball_params {};
    DepartureParams dep_params {};
};

}
