#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

#include <stator/core/concepts.h>
#include <stator/core/constants.h>
#include <stator/core/types.h>
#include <stator/core/errors.h>

using namespace stator::core;

namespace stator::physics {

// Lap Durations
// dT[j] is the lap FROM crossing j TO crossing j+1
inline std::vector<real> lap_durations(Params tk)
{
    if (tk.size() < 2)
        throw InvalidArgument("lap_durations(): tk must have at least 2 crossings, got {}", tk.size());

    const idx nlap { tk.size() - 1 };
    std::vector<real> dT(nlap, 0.0);
    for (idx j {}; j < nlap; j++)
    {
        dT[j] = tk[j+1] - tk[j];
        if (dT[j] <= 0.0)
            throw InvalidArgument("lap_durations(): non-monotonic crossings at j={} (dT={})", j, dT[j]);
    }
    return dT;
}

// Curvature of the Estimator Basis
inline real lap_curvature(Params dT, const real lap_floor)
{
    const idx ndat { dT.size() };
    if (ndat < 3)
        throw InvalidArgument("lap_curvature(): need at least 3 laps to fit a quadratic, got {}", ndat);

    // Centring kills the odd moments (sum(u) = sum(u^3) = 0 on a regular grid)
    const real ubar { 0.5*static_cast<real>(ndat - 1) };
    real S0 {}, S2 {}, S4 {}, b0 {}, b1 {}, b2 {};
    for (idx j {}; j < ndat; j++)
    {
        const real y { std::pow(2*pi/dT[j], 2) - lap_floor };
        if (y <= 0.0) return std::numeric_limits<real>::quiet_NaN();

        const real z { std::log(y) };
        const real u { static_cast<real>(j) - ubar };
        const real u2 { u*u };
        S0 += 1.0;
        S2 += u2;
        S4 += u2*u2;
        b0 += z;
        b1 += u*z;
        b2 += u2*z;
    }

    // Normal equations, odd moments dropped
    const real det { S0*S4 - S2*S2 };
    if ((det == 0.0) || (S2 == 0.0)) return std::numeric_limits<real>::quiet_NaN();
    const real c1 { b1 / S2 };                    // slope at the window centre
    const real c2 { (S0*b2 - S2*b0) / det };      // curvature
    if (c1 == 0.0) return std::numeric_limits<real>::quiet_NaN();

    return std::abs(c2 / c1);
}

// Physics-Basis To Estimator
inline real estimate_To(Params tk,
                        const idx m,
                        const real lap_floor,
                        const real a_slope,
                        const idx N)
{
    // Argument Checks
    if (N < 1)
        throw InvalidArgument("estimate_To(): window N must be at least 1");
    if (m < 1)
        throw InvalidArgument("estimate_To(): anchor m={} needs at least one observed lap before it", m);

    const std::vector<real> dT { lap_durations(tk) };

    // // Window: the N most recent observed laps at or before the anchor
    const idx hi { std::min(m - 1, dT.size() - 1) };
    if (hi + 1 < N)
        throw InvalidArgument("estimate_To(): {} laps available before anchor m={}, need N={}",
                              hi + 1, m, N);
    const idx lo { hi + 1 - N };

    // // Fit the Intercept (slope frozen)
    const real slope { -4.0 * pi * a_slope };
    real acc {};
    for (idx j {lo}; j <= hi; j++)
    {
        const real omega_sq { std::pow(2*pi/dT[j], 2) };
        const real y { omega_sq - lap_floor };
        if (y <= 0.0)
            throw InvalidArgument("estimate_To(): lap_floor={} is not below (2*pi/dT)^2={} at j={}",
                                  lap_floor, omega_sq, j);
        acc += std::log(y) - slope*static_cast<real>(j);
    }
    const real z0 { acc / static_cast<real>(N) };

    // // Evaluate at the Anchor
    const real z_m { z0 + slope*static_cast<real>(m) };
    const real w { std::exp(z_m) + lap_floor };
    if (w <= 0.0)
        throw InvalidArgument("estimate_To(): non-positive Omega^2={} at anchor m={}", w, m);

    return 2*pi / std::sqrt(w);
}

// Fit the Smoother's Lap Floor
inline real fit_lap_floor(const std::vector<std::vector<real>>& tks,
                          const idx N,
                          const idx N_scan = 60)
{
    // Argument Checks
    if (tks.empty())
        throw InvalidArgument("fit_lap_floor(): no spins supplied");
    if (N < 3)
        throw InvalidArgument("fit_lap_floor(): window N must be at least 3 to fit a quadratic, got {}", N);
    if (N_scan < 2)
        throw InvalidArgument("fit_lap_floor(): N_scan must be at least 2, got {}", N_scan);

    // // Gather the trailing window of every usable spin, and the domain ceiling
    std::vector<std::vector<real>> windows;
    real y_min { std::numeric_limits<real>::infinity() };
    for (const std::vector<real>& tk : tks)
    {
        if (tk.size() < N + 1) continue;    // fewer than N laps
        const std::vector<real> dT { lap_durations(tk) };
        std::vector<real> window(dT.end() - static_cast<std::ptrdiff_t>(N), dT.end());
        for (const real d : window)
            y_min = std::min(y_min, std::pow(2*pi/d, 2));
        windows.push_back(std::move(window));
    }
    if (windows.empty())
        throw InvalidArgument("fit_lap_floor(): no spin has the {} laps required by the window", N);

    // // Sweep for minimum total curvature
    const real floor_lo { -y_min };
    const real floor_hi { 0.95*y_min };
    const real step { (floor_hi - floor_lo) / static_cast<real>(N_scan - 1) };

    real best_floor {};
    real best_curv { std::numeric_limits<real>::infinity() };
    for (idx i {}; i < N_scan; i++)
    {
        const real lap_floor { floor_lo + static_cast<real>(i)*step };

        real curv {};
        bool in_domain { true };
        for (const std::vector<real>& window : windows)
        {
            const real c { lap_curvature(window, lap_floor) };
            if (!std::isfinite(c)) { in_domain = false; break; }
            curv += c;
        }

        if (in_domain && (curv < best_curv))
        {
            best_curv = curv;
            best_floor = lap_floor;
        }
    }
    if (!std::isfinite(best_curv))
        throw ConvergenceFailure("fit_lap_floor(): no candidate floor stayed in domain on every spin");

    return best_floor;
}

}
