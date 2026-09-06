#pragma once

#include <cmath>
#include <vector>
#include <string>
#include <optional>
#include <algorithm>

#include <stator/core/concepts.h>
#include <stator/core/constants.h>
#include <stator/core/errors.h>
#include <stator/core/fitmed.h>
#include <stator/core/fitmrq.h>
#include <stator/core/rtsafe.h>
#include <stator/core/golden_section.h>
#include <stator/core/numeric_result.h>
#include <stator/core/utility.h>

#include <stator/physics/constants.h>
#include <stator/physics/types.h>

using namespace stator::core;

namespace stator::physics {


class Rim
{
public:
    // Constructors
    explicit Rim(real a, real b, real phi, real nu, real omega_sq)
        : m_params { FitParams(BallParams(a, b), DepartureParams(phi, nu, omega_sq) ) }
        , m_data { std::vector<BallTiming>{} }
    {}
    Rim() = default;

    // Fitting
    void fit()
    {
        if (m_data.empty())
            return;
        
    }

    // Utility
    void add_timing(std::string_view id, std::vector<real>& timestamps, real theta)
    {
        if (timestamps.size() <= 2)
            throw InvalidArgument("Rim.add_timing(): timing ID {} must have more than two timestamps.", id);
        if (theta > 2*pi)
            throw InvalidArgument("Rim.add_timing(): timing ID {} departure angle theat exceeds 2pi", id, theta);
        for (idx j {1}; j < timestamps.size(); j++)
            if (timestamps[j] <= timestamps[j-1])
                throw InvalidArgument("Rim.add_timing(): timing ID {} has non-monotonic timestamps (t[{}] <= t[{}])", id, j, j-1);
        m_data.emplace_back(id, timestamps, theta);
    }

    void add_timing(std::vector<std::string>& ids, std::vector<std::vector<real>>& timestamps, std::vector<real>& thetas)
    {
        if ((ids.size() != timestamps.size()) || (timestamps.size() != thetas.size()))
            throw InvalidArgument("Rim.add_timing(): size of ids, timestamps, and bins vectors are not equal.");
        for (idx j {}; j < ids.size(); j++) 
            add_timing(ids[j], timestamps[j], thetas[j]);
    }


private:
    // Ball/Rim Phyiscal Parameters
    FitParams m_params {};

    // Consts

    // Data
    std::vector<BallTiming> m_data;

    // Private Functions
    idx ndat() const {return m_data.size(); };
};


}