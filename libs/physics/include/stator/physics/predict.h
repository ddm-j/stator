#pragma once

#include <vector>
#include <algorithm>
#include <cmath>

#include <stator/core/concepts.h>
#include <stator/core/constants.h>
#include <stator/core/types.h>
#include <stator/core/errors.h>
#include <stator/core/utility.h>
#include <stator/core/numeric_result.h>
#include <stator/core/rtsafe.h>

#include <stator/physics/types.h>

using namespace stator::core;

namespace stator::physics {

inline std::optional<real> predict_theta(const real To, const FitParams& params)
{
    // Unpack Fitted Params
    const real a { params.ball_params.a };
    const real b { params.ball_params.b };
    const real phi { params.dep_params.phi };
    const real eta { params.dep_params.eta };
    const real omega_sq { params.dep_params.omega_sq };

    // Useful Shortcuts
    const real A { 1.5 + 2 * a * a };
    const real M { std::abs(eta) * std::sqrt(A*A + 4*a*a)};
    const real K { omega_sq - b*b };
    if (M+K <= 0) return std::nullopt; // Ball never leaves the track

    // Initial Conditions
    const real x  { (std::exp(2*pi*a) - std::cosh(a*b*To)) / std::sinh(a*b*To) };
    const real c1 { b*b*(x*x - 1) };
    if (c1 <= 0) return std::nullopt; // Invalid starting condition

    // The Root Finding target
    auto g = [c1, a, A, K, eta, phi] (real theta) -> real {
        const real D { c1 * std::exp(-2.0*a*theta) - K };
        const real O { eta * (A*std::cos(theta+phi) - 2*a*std::sin(theta+phi)) };
        return D + O;
    };

    // Bounding Theta
    const real theta_lo { std::max(0.0, -std::log((M+K)/c1) / (2*a)) };
    const real theta_hi  { theta_lo + 2*pi };
    
    // Check Before Loop
    if (g(theta_lo) < 0.0) return theta_lo;    // Tangency
    if (g(theta_lo) == 0.0) return std::nullopt; // Ball already left track (inconsistency in observations)

    // Root Finding (g(theta) = 0)
    real prev {theta_lo};
    const real N { 64.0 };
    const real h { 2*pi/N };
    for (real cur {theta_lo + h}; cur <= (N*real_EPS + theta_hi); cur += h)
    {
        const real g_cur { g(cur) };
        if (g_cur <= 0.0)
        {
            RootResult res { rtsafe(g, prev, cur) };
            if (!res.converged) return std::nullopt; // root find failed
            return res.x;
        }
        prev = cur;
    }
    return std::nullopt; // no root found over [theta_hi, theta_lo]
}

}