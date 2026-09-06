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
#include <stator/core/fitmrq.h>
#include <stator/core/golden_section.h>

#include <stator/physics/constants.h>
#include <stator/physics/types.h>

using namespace stator::core;

namespace stator::physics {

inline real acoth(real x)
{
    // Is allowed to return a NaN
    return 0.5 * std::log((x + 1.0)/(x - 1.0));
}

// Ball Timing Model
inline real ball_timing_model(real k, Params p)
{
    const real a { p[0] };
    const real b { p[1] };
    const real co { p[2] };
    const real f { 
        (1.0 / (a * b)) * (co - std::asinh(std::sinh(co) * std::exp(2*a*k*pi)))
    };
    return f;
}

// Ball Lap Timing Refinement Model tunes a/b subject to beta = C = ab^2
// Compliant with concepts/Model
inline auto make_ball_timing_refinement_model(const real beta, Params t_k)
{
    if (t_k.size() < 2)
        throw InvalidArgument("make_ball_timing_refinement_model(): t_k must have at least 2 measurements");

    // Model Constants
    const idx ndat { t_k.size() };
    const real To { t_k[1] - t_k[0] };

    // Take a copy of t_k so that lifetimes are managed properly
    const std::vector<real> tt_k(t_k.begin(), t_k.end());

    auto ball_timing_refinement_model = [tt_k, beta, To, ndat](real a) -> real {
        const real ab { std::sqrt(beta*a) };
        const real x { (std::exp(2*a*pi) - std::cosh(ab*To)) / std::sinh(ab*To) };
        const real co { -acoth(x) };
        real S {};
        for (idx j {}; j < ndat; j++)
        {
            const real k { static_cast<real>(j) };
            const real f { 
                (1.0 / ab) * (co - std::asinh(std::sinh(co) * std::exp(2*a*k*pi)))
            };
            S += std::pow(tt_k[j] - f, 2);
        }
        return S;
    };
    return ball_timing_refinement_model;
}

inline FitResult fit_ball_timings(Params tk,
                                 const real sig,
                                 const real a = a_i,
                                 const real b = b_i,
                                 const real c0 = c0_i)
{
    // Setup Timings
    std::vector<real> ttk {tk.begin(), tk.end()};
    std::vector<real> k(tk.size(), 0.0);
    for (idx j {}; j < tk.size(); j++)
        k[j] = static_cast<real>(j);

    // Parameter Vector
    const std::vector<real> p { a, b, c0 };

    // Solve
    FitResult res { fitmrq(ball_timing_model, k, ttk, sig, p) };

    return res;
}


inline BallParams fit_ab(const std::vector<real>& tk)
{
    // Levenberg-Marquardt Fit of A and B
    // LM Fit
    FitResult res { fit_ball_timings(tk, sig_human)};
    return {res.a[0], res.a[1]};
}

inline BallParams refine_ab(const BallParams ball_params, const std::vector<real>& tk)
{
    // Gold Sec Refinement
    const real beta { ball_params.a*std::pow(ball_params.b, 2) }; // beta = ab^2
    if (beta <= 0.0)
        throw InvalidArgument("refine_ab(): invalid ball parameters. a*b^2 is <= 0");

    auto refinement_model = make_ball_timing_refinement_model(beta, tk);

    // // Bound A
    real dT_max {};
    for (idx j {1}; j < tk.size(); j++)
        dT_max = std::max(dT_max, tk[j] - tk[j-1]);
    const real a_lo { 1.01 * beta * dT_max*dT_max / (4*pi*pi) };
    const real a_hi { 3.0 * ball_params.a };
    if (ball_params.b * dT_max >= 2*pi)
        throw InvalidArgument("refine_ab(): b={} exceeds 2*pi/dT_max={}; "
                              "inconsistent with observed lap times",
                              ball_params.b, 2*pi/dT_max);

    // // Scan S(a) to form a valid bracket
    const idx N_scan {30};
    std::vector<real> as { logspace(std::log10(a_lo), std::log10(a_hi), N_scan) };
    std::vector<real> S(N_scan, 0.0);
    for (idx j {}; j < N_scan; j++) S[j] = refinement_model(as[j]);
    const idx m { argmin(S) };
    idx l {m}; while (l > 0 && S[l] <= S[m]) --l;
    idx r {m}; while (r < N_scan-1 && S[r] <= S[m]) ++r;

    // // Minimize over the bracket
    Bracket brack { as[l], as[m], as[r], S[l], S[m], S[r] };
    if (!brack.is_valid())
    {
        if (brack.fa >= brack.fc)
            return {brack.ax, std::sqrt(beta/brack.fa)};
        else
            return {brack.cx, std::sqrt(beta/brack.fc)};
    }
    MinResult1D min { golden_section(refinement_model, brack) };

    // Unpack
    const real a_new { min.x };
    const real b_new { std::sqrt(beta/min.x) };

    return {a_new, b_new};
}

}