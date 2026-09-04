#pragma once

#include <cmath>
#include <vector>

#include <stator/core/concepts.h>
#include <stator/core/constants.h>
#include <stator/core/errors.h>
#include <stator/core/fitmed.h>
#include <stator/core/fitmrq.h>
#include <stator/core/numeric_result.h>
#include <stator/physics/constants.h>

using namespace stator::core;

namespace stator::physics {

inline real acoth(real x)
{
    // Is allowed to return a NaN
    return 0.5 * std::log((x + 1.0)/(x - 1.0));
}

// Ball Lap Timing Model
// Compliant with concepts/Model
inline real ball_timing_model(real k, Params params)
{
    if (params.size() != 3)
        throw InvalidArgument("ball_timing_model(): Wrong number of parameters. Takes three: a, b, c0");

    const real a { params[0] };
    const real b { params[1] };
    const real co { params[2] };
    const real f { 
        (1.0 / (a * b)) * (co - std::asinh(std::sinh(co) * std::exp(2*a*k*pi)))
    };
    return f;
};

// Ball Lap Timing Refinement Model
// Compliant with concepts/Model
inline auto make_ball_timing_refinement_model(const real b, Params t_k)
{
    if (t_k.size() < 2)
        throw InvalidArgument("make_ball_timing_refinement_model(): t_k must have at least 2 measurements");

    // Model Constants
    const idx ndat { t_k.size() };
    const real To { t_k[1] - t_k[0] };

    // Take a copy of t_k so that lifetimes are managed properly
    const std::vector<real> tt_k(t_k.begin(), t_k.end());

    auto ball_timing_refinement_model = [tt_k, b, To, ndat](real a) -> real {
        const real x { (std::exp(2*a*pi) - std::cosh(a*b*To)) / std::sinh(a*b*To) };
        const real co { -acoth(x) };
        const std::vector<real> timing_params { a, b, co };
        real S {};
        for (idx j {}; j < ndat; j++)
        {
            const real k { static_cast<real>(j) };
            const real f { ball_timing_model(k, timing_params )};
            S += std::pow(tt_k[j] - f, 2);
        }
        return S;
    };
    return ball_timing_refinement_model;
}

// Ball Departure Objective Function Factory
inline auto make_departure_objective(const real a, const real b, Params To, Params theta_f)
{
    if (To.size() != theta_f.size())
        throw InvalidArgument("make_departure_objective(): To.size() must equal theta_f.size()");
    
    // Setup
    const idx ndat { To.size() };
    const std::vector<real> ttheta_f { theta_f.begin(), theta_f.end() };
    std::vector<real> x(ndat, 0.0);
    std::vector<real> c1(ndat, 0.0);
    std::vector<real> yk(ndat, 0.0);
    for (idx j {}; j < ndat; j++)
    {
        x[j] = (std::exp(2.0*a*pi) - std::cosh(a*b*To[j])) / std::sinh(a*b*To[j]);
        c1[j] = b*b*(x[j]*x[j] - 1);
        yk[j] = c1[j]*std::exp(-2.0*a*theta_f[j]) + b*b;
    }

    // The Objective
    const real A { 1.0 + 0.5*(4.0*a*a + 1.0) };
    auto departure_objective = [a, A, ndat, yk, ttheta_f] (real phi) {
        std::vector<real> Xk(ndat, 0.0);
        for (idx j {}; j < ndat; j++)
            Xk[j] = A * std::cos(ttheta_f[j] + phi) - 2.0*a*std::sin(ttheta_f[j] + phi);
        
        // L1 Fit
        const LinRegResult res { fitmed(Xk, yk) };
        return res.abdev;
    };
    return departure_objective;
}

inline FitResult fit_ball_timing(Params tk,
                                 const real sig,
                                 const real a = a_i,
                                 const real b = b_i,
                                 const real co = co_i)
{
    // Setup Timings subtract t0
    std::vector<real> ttk {tk.begin(), tk.end()};
    std::vector<real> k(tk.size(), 0.0);
    for (idx j {}; j < tk.size(); j++)
    {
        ttk[j] -= - tk[0];
        k[j] = static_cast<real>(j);
    }

    // Parameter Vector
    const std::vector<real> p { a, b, co };

    return fitmrq(ball_timing_model, k, ttk, sig, p);
}


}