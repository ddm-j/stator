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
        return fitmed(Xk, yk);
    };
    return departure_objective;
}

// Overload for per-spin a/b
inline auto make_departure_objective(Params a, Params b, Params To, Params theta_f)
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
        x[j] = (std::exp(2.0*a[j]*pi) - std::cosh(a[j]*b[j]*To[j])) / std::sinh(a[j]*b[j]*To[j]);
        c1[j] = b[j]*b[j]*(x[j]*x[j] - 1);
        yk[j] = c1[j]*std::exp(-2.0*a[j]*theta_f[j]) + b[j]*b[j];
    }

    // The Objective
    auto departure_objective = [a, ndat, yk, ttheta_f] (real phi) {
        std::vector<real> Xk(ndat, 0.0);
        for (idx j {}; j < ndat; j++)
        {
            const real A { 1.0 + 0.5*(4.0*a[j]*a[j] + 1.0) };
            Xk[j] = A * std::cos(ttheta_f[j] + phi) - 2.0*a[j]*std::sin(ttheta_f[j] + phi);
        }
        
        // L1 Fit
        return fitmed(Xk, yk);
    };
    return departure_objective;
}

inline DepartureParams fit_departure(const std::vector<real>& To, const std::vector<real>& theta, const BallParams ball_params)
{
    // Argument Check
    if (To.size() != theta.size())
        throw InvalidArgument("fit_departure(): To.size() and theta.size() must be equal.");

    // Fit Departure
    auto departure_model = make_departure_objective(ball_params.a, ball_params.b, To, theta);

    /// // Coarse Scan to find global minimum
    const idx N {30};
    const real dphi { 2*pi/static_cast<real>(N) };
    std::vector<real> S_coarse(N, 0.0);
    for (idx i {}; i < N; i++)
    {
        real phi { static_cast<real>(i) * dphi };
        S_coarse[i] = departure_model(phi).abdev;
    }
    auto min_it = std::min_element(S_coarse.begin(), S_coarse.end());
    idx idx_min { static_cast<idx>(std::distance(S_coarse.begin(), min_it)) };

    // // Hone the First Minimum
    const MinResult1D min1_res { golden_section([&](real phi) { return departure_model(phi).abdev; }, 
                                                static_cast<real>(idx_min)*dphi - dphi, 
                                                static_cast<real>(idx_min)*dphi + dphi) };
    real phi1 { min1_res.x };

    // // Second Minimum (Invariant of the Model)
    real phi2 { phi1 + pi };

    // Determine Which Phi is best 
    phi1 = std::fmod(phi1, 2*pi);
    phi2 = std::fmod(phi2, 2*pi);
    std::pair<LinRegResult, LinRegResult> phi_fits { departure_model(phi1), departure_model(phi2) };
    if ((phi_fits.first.b > 0) && (phi_fits.second.b > 0))
        throw InvalidArgument("No positive nu");
    
    real phi {};
    real eta {};
    real omega_fsq {};
    if (phi_fits.first.b < phi_fits.second.b)
    {
        phi = phi1;
        eta = -phi_fits.first.b;
        omega_fsq = phi_fits.first.a;
    } else 
    {
        phi = phi2;
        eta = -phi_fits.second.b;
        omega_fsq = phi_fits.second.a;
    }
    return {phi, eta, omega_fsq};
}

inline DepartureParams fit_departure_perspin(const std::vector<std::vector<real>>& tks, const std::vector<real>& To, const std::vector<real>& theta, const BallParams ball_params)
{
    // Argument Check
    if (
        (tks.size() != To.size()) ||
        (To.size() != theta.size())
    )
        throw InvalidArgument("fit_departure_perspin(): Invalid vector sizes. Size of tks, To, and theta must match.");

    // Refine per-spin a/b
    std::vector<real> as(To.size(), ball_params.a);
    std::vector<real> bs(To.size(), ball_params.b);
    for (idx j {}; j < To.size(); j++)
    {
        BallParams ref_params { refine_ab(ball_params, tks[j]) };
        as[j] = ref_params.a;
        bs[j] = ref_params.b;
    }

    // Fit Departure
    auto departure_model = make_departure_objective(as, bs, To, theta);

    /// // Coarse Scan to find global minimum
    const idx N {30};
    const real dphi { 2*pi/static_cast<real>(N) };
    std::vector<real> S_coarse(N, 0.0);
    for (idx i {}; i < N; i++)
    {
        real phi { static_cast<real>(i) * dphi };
        S_coarse[i] = departure_model(phi).abdev;
    }
    auto min_it = std::min_element(S_coarse.begin(), S_coarse.end());
    idx idx_min { static_cast<idx>(std::distance(S_coarse.begin(), min_it)) };

    // // Hone the First Minimum
    const MinResult1D min1_res { golden_section([&](real phi) { return departure_model(phi).abdev; }, 
                                                static_cast<real>(idx_min)*dphi - dphi, 
                                                static_cast<real>(idx_min)*dphi + dphi) };
    real phi1 { min1_res.x };

    // // Second Minimum Invariant
    real phi2 { phi1 + pi };

    // Determine Which Phi is best 
    phi1 = std::fmod(phi1, 2*pi);
    phi2 = std::fmod(phi2, 2*pi);
    std::pair<LinRegResult, LinRegResult> phi_fits { departure_model(phi1), departure_model(phi2) };
    if ((phi_fits.first.b > 0) && (phi_fits.second.b > 0))
        throw InvalidArgument("No positive nu");
    
    real phi {};
    real eta {};
    real omega_fsq {};
    if (phi_fits.first.b < phi_fits.second.b)
    {
        phi = phi1;
        eta = -phi_fits.first.b;
        omega_fsq = phi_fits.first.a;
    } else 
    {
        phi = phi2;
        eta = -phi_fits.second.b;
        omega_fsq = phi_fits.second.a;
    }
    return {phi, eta, omega_fsq};
}
}