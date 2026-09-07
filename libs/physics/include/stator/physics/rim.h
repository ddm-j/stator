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

#include <stator/physics/ball.h>
#include <stator/physics/departure.h>
#include <stator/physics/timing.h>
#include <stator/physics/predict.h>
#include <stator/physics/constants.h>
#include <stator/physics/types.h>

using namespace stator::core;

namespace stator::physics {


class Rim
{
public:
    // Constructors
    explicit Rim(real a, real b, real delta, real eta, real omega_sq)
        : m_params { FitParams(BallParams(a, b), DepartureParams(delta, eta, omega_sq) ) }
        , m_data { std::vector<BallTiming>{} }
    {}
    Rim() = default;
    Rim(idx M, idx Y)
        : m_M { M }
        , m_Y { Y }
    {}

    // Fitting
    // M laps of lead time between the anchor and the drop, Y laps of To window
    // before the anchor. a/b are fit on everything ahead of the To window, so
    // each stage sees only what it would see live.
    void fit()
    {
        if (m_data.empty())
            return;

        // Ball Parameters, per spin
        std::vector<real> as, bs, thetas, ss;
        std::vector<std::vector<real>> tks;
        std::vector<idx> anchors;
        for (const BallTiming& spin : m_data)
        {
            const idx n { spin.tk.size() };
            if (n < m_M + m_Y + 1) continue;    // anchor must clear the To window

            const idx anchor { n - 1 - m_M };
            const auto beg { spin.tk.begin() };

            const BallParams p { fit_ab(spin.tk) };         // pooled constant, use every lap
            if ((p.a <= 0.0) || (p.b == 0.0)) continue;     // degenerate fit

            as.push_back(p.a);
            bs.push_back(std::abs(p.b));                    // sign of b is not identified
            tks.emplace_back(beg, beg + static_cast<std::ptrdiff_t>(anchor) + 1);
            anchors.push_back(anchor);
            thetas.push_back(spin.theta);
            ss.push_back(spin.s);
        }
        if (as.empty())
            throw ConvergenceFailure("Rim.fit(): no spin produced usable ball parameters");

        // Pool
        const BallParams ball { detail::median(as), detail::median(bs) };

        // To Estimator Constants
        m_lap_floor = fit_lap_floor(tks, m_Y);
        m_a_slope = ball.a;

        // To and Departure Angle, both referenced to the anchor
        std::vector<real> To, theta;
        for (idx j {}; j < tks.size(); j++)
        {
            To.push_back(estimate_To(tks[j], anchors[j], m_lap_floor, m_a_slope, m_Y));
            theta.push_back(thetas[j] - 2*pi*static_cast<real>(anchors[j]));
        }

        // Fit Departure
        m_params = { ball, fit_departure(To, theta, ss, ball) };
    }

    // Prediction
    // tk's LAST crossing is the anchor. To is the lap the ball is on now,
    // extrapolated from the Y laps behind it, matching how fit() built To[].
    std::optional<BallPrediction> predict(const std::vector<real>& tk, const real s = 1.0) const
    {
        if (m_params.ball_params.a <= 0.0)
            throw InvalidArgument("Rim.predict(): rim has not been fit");

        const idx n { tk.size() };
        if (n < m_Y + 1)
            throw InvalidArgument("Rim.predict(): need {} crossings for a {} lap window, got {}",
                                  m_Y + 1, m_Y, n);

        // Trim off the front, the To estimator only wants the last Y laps
        const std::vector<real> tk_win(tk.end() - static_cast<std::ptrdiff_t>(m_Y) - 1, tk.end());
        const idx m { tk_win.size() - 1 };

        const real To { estimate_To(tk_win, m, m_lap_floor, m_a_slope, m_Y) };
        const std::optional<real> theta { predict_theta(To, s, m_params) };
        if (!theta)
            return std::nullopt;

        return BallPrediction { *theta, predict_tf(To, *theta, s, m_params) };
    }

    std::vector<std::optional<BallPrediction>> predict(const std::vector<std::vector<real>>& tks,
                                                       const std::vector<real>& ss = {}) const
    {
        if (!ss.empty() && (ss.size() != tks.size()))
            throw InvalidArgument("Rim.predict(): ss must be empty or match tks in size");

        std::vector<std::optional<BallPrediction>> res;
        res.reserve(tks.size());
        for (idx j {}; j < tks.size(); j++)
            res.push_back(predict(tks[j], ss.empty() ? 1.0 : ss[j]));
        return res;
    }

    // Accessors
    const FitParams& params() const { return m_params; }
    real lap_floor() const { return m_lap_floor; }
    real a_slope() const { return m_a_slope; }

    // Utility
    void add_timing(std::string_view id, std::vector<real>& timestamps, real theta, real s = 1.0)
    {
        if ((s != 1.0) && (s != -1.0))
            throw InvalidArgument("Rim.add_timing(): timing ID {} direction s must be +1 or -1, got {}", id, s);
        if (timestamps.size() <= 2)
            throw InvalidArgument("Rim.add_timing(): timing ID {} must have more than two timestamps.", id);
        if (theta > 2*pi)
            throw InvalidArgument("Rim.add_timing(): timing ID {} departure angle theat exceeds 2pi", id, theta);
        for (idx j {1}; j < timestamps.size(); j++)
            if (timestamps[j] <= timestamps[j-1])
                throw InvalidArgument("Rim.add_timing(): timing ID {} has non-monotonic timestamps (t[{}] <= t[{}])", id, j, j-1);
        m_data.emplace_back(id, timestamps, theta, s);
    }

    void add_timing(std::vector<std::string>& ids, std::vector<std::vector<real>>& timestamps,
                    std::vector<real>& thetas, std::vector<real> ss = {})
    {
        if ((ids.size() != timestamps.size()) || (timestamps.size() != thetas.size()))
            throw InvalidArgument("Rim.add_timing(): size of ids, timestamps, and bins vectors are not equal.");
        if (!ss.empty() && (ss.size() != ids.size()))
            throw InvalidArgument("Rim.add_timing(): ss must be empty or match ids in size.");
        for (idx j {}; j < ids.size(); j++)
            add_timing(ids[j], timestamps[j], thetas[j], ss.empty() ? 1.0 : ss[j]);
    }


private:
    // Ball/Rim Phyiscal Parameters
    FitParams m_params {};

    // Consts
    idx m_M { 4 };              // lead laps between the anchor and the drop
    idx m_Y { 6 };              // To estimator window, in laps
    real m_lap_floor {};
    real m_a_slope {};

    // Data
    std::vector<BallTiming> m_data;

    // Private Functions
    idx ndat() const {return m_data.size(); };
};


}