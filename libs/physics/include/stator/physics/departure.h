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
#include <stator/core/fitmed.h>
#include <stator/core/golden_section.h>

#include <stator/physics/constants.h>
#include <stator/physics/types.h>
#include <stator/physics/ball.h>


namespace stator::physics {

// Ball Departure Objective Function Factory
// delta is the lab frame angle between the reference mark and the wheel's low
// point. It is direction blind. The travel frame phase the paper's eq (37)
// wants is phi = s*delta, since reversing the ball reverses the sense in which
// that same physical angle is measured.
inline auto make_departure_objective(const real a, const real b, Params To, Params theta_f, Params s)
{
    if ((To.size() != theta_f.size()) || (To.size() != s.size()))
        throw InvalidArgument("make_departure_objective(): To, theta_f and s must be the same size");

    // Setup
    const idx ndat { To.size() };
    const std::vector<real> ttheta_f { theta_f.begin(), theta_f.end() };
    const std::vector<real> ss { s.begin(), s.end() };
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
    auto departure_objective = [a, A, ndat, yk, ttheta_f, ss] (real delta) {
        std::vector<real> Xk(ndat, 0.0);
        for (idx j {}; j < ndat; j++)
        {
            const real phi { ss[j]*delta };
            Xk[j] = A * std::cos(ttheta_f[j] + phi) - 2.0*a*std::sin(ttheta_f[j] + phi);
        }

        // L1 Fit
        return fitmed(Xk, yk);
    };
    return departure_objective;
}

// Overload for per-spin a/b
inline auto make_departure_objective(Params a, Params b, Params To, Params theta_f, Params s)
{
    if ((To.size() != theta_f.size()) || (To.size() != s.size()))
        throw InvalidArgument("make_departure_objective(): To, theta_f and s must be the same size");

    // Setup
    const idx ndat { To.size() };
    const std::vector<real> ttheta_f { theta_f.begin(), theta_f.end() };
    const std::vector<real> ss { s.begin(), s.end() };
    std::vector<real> x(ndat, 0.0);
    std::vector<real> c1(ndat, 0.0);
    std::vector<real> yk(ndat, 0.0);
    for (idx j {}; j < ndat; j++)
    {
        x[j] = (std::exp(2.0*a[j]*pi) - std::cosh(a[j]*b[j]*To[j])) / std::sinh(a[j]*b[j]*To[j]);
        c1[j] = b[j]*b[j]*(x[j]*x[j] - 1);
        yk[j] = c1[j]*std::exp(-2.0*a[j]*theta_f[j]) + b[j]*b[j];
    }

    // The Objective
    auto departure_objective = [a, ndat, yk, ttheta_f, ss] (real delta) {
        std::vector<real> Xk(ndat, 0.0);
        for (idx j {}; j < ndat; j++)
        {
            const real A { 1.0 + 0.5*(4.0*a[j]*a[j] + 1.0) };
            const real phi { ss[j]*delta };
            Xk[j] = A * std::cos(ttheta_f[j] + phi) - 2.0*a[j]*std::sin(ttheta_f[j] + phi);
        }

        // L1 Fit
        return fitmed(Xk, yk);
    };
    return departure_objective;
}

// Minimize a departure objective over delta
// The objective has two minima of identical value, pi apart, because
// delta -> delta+pi flips Xk for every spin whatever its direction. Only the
// sign of the slope distinguishes them, and the physical branch is eta > 0.
template <typename F>
inline DepartureParams solve_departure(F&& departure_model)
{
    // // Coarse Scan to find global minimum
    const idx N {30};
    const real ddelta { 2*pi/static_cast<real>(N) };
    std::vector<real> S_coarse(N, 0.0);
    for (idx i {}; i < N; i++)
        S_coarse[i] = departure_model(static_cast<real>(i) * ddelta).abdev;
    const idx idx_min { argmin(S_coarse) };

    // // Hone the First Minimum
    const MinResult1D min1_res { golden_section([&](real d) { return departure_model(d).abdev; },
                                                static_cast<real>(idx_min)*ddelta - ddelta,
                                                static_cast<real>(idx_min)*ddelta + ddelta) };
    real delta1 { min1_res.x };

    // // Hone the Second Minimum
    const MinResult1D min2_res { golden_section([&](real d) { return departure_model(d).abdev; },
                                                delta1 + pi - ddelta,
                                                delta1 + pi + ddelta) };
    real delta2 { min2_res.x };

    // // Determine Which Delta is best
    delta1 = std::fmod(delta1, 2*pi);
    delta2 = std::fmod(delta2, 2*pi);
    const std::pair<LinRegResult, LinRegResult> fits { departure_model(delta1), departure_model(delta2) };
    if ((fits.first.b > 0) && (fits.second.b > 0))
        throw InvalidArgument("solve_departure(): no positive eta");

    if (fits.first.b < fits.second.b)
        return {delta1, -fits.first.b, fits.first.a};
    return {delta2, -fits.second.b, fits.second.a};
}

inline DepartureParams fit_departure(const std::vector<real>& To, const std::vector<real>& theta,
                                     const std::vector<real>& s, const BallParams ball_params)
{
    // Argument Check
    if ((To.size() != theta.size()) || (To.size() != s.size()))
        throw InvalidArgument("fit_departure(): To, theta and s must be the same size.");

    return solve_departure(make_departure_objective(ball_params.a, ball_params.b, To, theta, s));
}

inline DepartureParams fit_departure_perspin(const std::vector<std::vector<real>>& tks,
                                             const std::vector<real>& To, const std::vector<real>& theta,
                                             const std::vector<real>& s, const BallParams ball_params)
{
    // Argument Check
    if (
        (tks.size() != To.size()) ||
        (To.size() != theta.size()) ||
        (To.size() != s.size())
    )
        throw InvalidArgument("fit_departure_perspin(): Invalid vector sizes. Size of tks, To, theta and s must match.");

    // Refine per-spin a/b
    std::vector<real> as(To.size(), ball_params.a);
    std::vector<real> bs(To.size(), ball_params.b);
    for (idx j {}; j < To.size(); j++)
    {
        BallParams ref_params { refine_ab(ball_params, tks[j]) };
        as[j] = ref_params.a;
        bs[j] = ref_params.b;
    }

    return solve_departure(make_departure_objective(as, bs, To, theta, s));
}

}
