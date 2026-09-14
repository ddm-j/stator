#pragma once

#include <vector>
#include <cmath>
#include <string>
#include <span>
#include <ranges>
#include <optional>

#include <linalg/solvers.h>

#include <stator/core/types.h>
#include <stator/core/errors.h>

#include <stator/physics/types.h>
#include <stator/physics/constants.h>

using namespace stator::core;

namespace stator::physics {

inline std::pair<real, real> fit_rotor(const std::vector<WheelTiming>& timings)
{
    // A super memory inneficient matrix solve to fit rotor deceleration
    // but uses my linalg library!

    using linalg::Matrix;

    // Index Constants
    const idx num_spins { timings.size() };
    const idx width { num_spins + 1 };
    const idx k_col { num_spins };
    idx num_rows {};
    for (idx i {}; i < timings.size(); i++)
        num_rows += timings[i].timestamps.size(); 

    // Create Matrix A and Vector y
    Matrix<real> A(num_rows, width);
    Matrix<real> y(num_rows, 1);
    idx row {};
    for (idx spin {}; spin < num_spins; spin++)
        for (idx sample {}; sample < timings[spin].timestamps.size(); sample++)
        {
            const real t { timings[spin].timestamps[sample] - timings[spin].timestamps[0] };
            if (t == 0.0) continue;

            A[row, spin] = t;
            A[row, k_col] = -t*t/2.0;
            y[row, 0] = timings[spin].angles[sample] - timings[spin].angles[0];
            ++row;
        }

    // Least Squares Solve
    Matrix<real> At = Matrix<real>(A.transpose());
    Matrix<real> AtA = At * A;
    Matrix<real> Aty = At * y;
    Matrix<real> p { linalg::solvers::solve_cholesky(AtA, Aty) };

    // The rotor decel constant 
    const real k { p[k_col, 0] };

    real sigma_k { std::numeric_limits<real>::quiet_NaN() };
    if ((row - width) == 0)
        return { k, sigma_k };

    // Measure k uncertainty
    Matrix<real> r { y - A*p };
    const real sig2 { (Matrix<real>(r.transpose()) * r)[0,0] / static_cast<real>(row-width) };
    Matrix<real> e_k(width, 1);
    e_k[k_col, 0] = 1.0;
    Matrix<real> z { linalg::solvers::solve_cholesky(AtA, e_k) };

    sigma_k = sqrt(sig2 * z[k_col, 0]);
    return {k, sigma_k/k};
};

class Wheel
{
public:
    static Wheel American() { return Wheel(american_wheel); };
    static Wheel European() { return Wheel(european_wheel); };

    // Fitting
    void fit()
    {
        if (m_wheel_timings.empty())
            return;
        
        auto [k, sig_k_k] = fit_rotor(m_wheel_timings);
        m_decay_k = k;
    }

    // Prediction
    real predict(std::vector<real>& timestamps, std::vector<real>& angles, const real t_drop) const
    {
        // Predicts the angle of the rotor at t_drop, in the same frame as angles
        if (m_decay_k == 0.0)
            throw InvalidArgument("Wheel.predict(): Wheel has not been fitted, decay constant == 0.");
        if (timestamps.size() != 2)
            throw InvalidArgument("Wheel.predict(): timestamps must be of size 2, representing the start and end of a rotor lap timing.");
        if (angles.size() != 2)
            throw InvalidArgument("Wheel.predict(): angles must be of size 2, matching timestamps.");
        if (timestamps[1] <= timestamps[0])
            throw InvalidArgument("Wheel.predict(): timestamps are non-monotonic. t[1] <= t[0]");
        if (angles[1] <= angles[0])
            throw InvalidArgument("Wheel.predict(): angles are non-monotonic. angle[1] <= angle[0]");
        if (t_drop <= timestamps[1])
            throw InvalidArgument("Wheel.predict(): t_drop must be a time after timestamps[1] in order to make a prediction");

        // Initial Angular Velocity
        const real T0 { timestamps[1] - timestamps[0] };
        const real v0 { (angles[1] - angles[0] + (m_decay_k / 2.0)*std::pow(T0, 2)) / T0};

        // Wheel Position Prediction
        const real t { t_drop - timestamps[0] };
        if (t > v0 / m_decay_k)
            throw NumericError("Wheel.predict(): prediction time horizon is beyond model capability. t_limit = {}s", v0/m_decay_k);
        const real theta { angles[0] + v0*t - (m_decay_k / 2.0) * std::pow(t, 2) };

        return theta;
    }

    // Utility
    void add_timing(std::string_view id, std::vector<real>& timestamps, std::vector<real>& angles)
    {
        if (timestamps.size() <= 2)
            throw InvalidArgument("Wheel.add_timing(): timing ID {} must have more than two timestamps.", id);
        if (angles.size() != timestamps.size())
            throw InvalidArgument("Wheel.add_timing(): timing ID {} has {} timestamps but {} angles.", id, timestamps.size(), angles.size());
        for (idx j {1}; j < timestamps.size(); j++)
        {
            if (timestamps[j] <= timestamps[j-1])
                throw InvalidArgument("Wheel.add_timing(): timing ID {} has non-monotonic timestamps (t[{}] <= t[{}])", id, j, j-1);
            if (angles[j] <= angles[j-1])
                throw InvalidArgument("Wheel.add_timing(): timing ID {} has non-monotonic angles (angle[{}] <= angle[{}])", id, j, j-1);
        }
        m_wheel_timings.emplace_back(id, timestamps, angles);
    }

    void add_timing(std::vector<std::string>& ids, std::vector<std::vector<real>>& timestamps, std::vector<std::vector<real>>& angles)
    {
        if ((ids.size() != timestamps.size()) || (timestamps.size() != angles.size()))
            throw InvalidArgument("Wheel.add_timing(): size of ids, timestamps, and angles vectors are not equal.");
        for (idx j {}; j < ids.size(); j++)
            add_timing(ids[j], timestamps[j], angles[j]);
    }

    // Accessors
    std::optional<idx> get_pkt_index(std::string_view s) const {
        const auto it = std::ranges::find(m_wheel_layout, s);
        if (it == m_wheel_layout.end()) return std::nullopt;
        return static_cast<idx>(it - m_wheel_layout.begin());
    }

    std::optional<real> get_pkt_angle(std::string_view s) const {
        std::optional<idx> i { get_pkt_index(s) };
        if (!i.has_value()) return std::nullopt;
        return index_to_angle(i.value());
    }

    std::string_view get_pkt_from_angle(real angle) const {
        // Modulo 2pi
        angle = std::fmod(angle, 2*pi);
        if (angle < 0.0) angle = 2*pi + angle;
        const idx index { static_cast<idx>(angle / m_pkt_ang) % m_n };
        return m_wheel_layout[index];
    }

    idx  get_n() const { return m_n; };
    real get_decay_k() const { return m_decay_k; };
    real get_pkt_ang() const { return m_pkt_ang; };

private:
    Wheel(std::span<const std::string_view> wheel_layout)
        : m_wheel_layout { wheel_layout }
        , m_n { m_wheel_layout.size() }
        , m_pkt_ang { 2*pi / static_cast<real>(m_n) }
        , m_wheel_timings {}
        , m_decay_k {}
    {}

    real index_to_angle(const idx index) const { return m_pkt_ang*(static_cast<real>(index) + 0.5); }

    std::span<const std::string_view> m_wheel_layout;
    const idx m_n {};
    const real m_pkt_ang {};
    std::vector<WheelTiming> m_wheel_timings {};
    real m_decay_k {};
};

}