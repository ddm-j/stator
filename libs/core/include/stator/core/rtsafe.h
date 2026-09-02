// rtsafe.h
#pragma once
#include <concepts>
#include <utility>
#include <tuple>
#include <cmath>
#include <iostream>

#include "stator/core/types.h"
#include "stator/core/constants.h"
#include "stator/core/concepts.h"
#include "stator/core/errors.h"
#include "stator/core/numeric_result.h"

namespace stator::core {

// rtsafe tolerance
real rtsafe_tol(const real rts, const real xacc)
{
    return 2.0 * std::fabs(rts) * real_EPS + 0.5*xacc;
}

// rtsafe routine (Numerical Recipes - Third Edition)
template <DifferentiableModel F>
RootResult rtsafe(F&& func, const real x1, const real x2, const real xacc = 0.0, const idx MAXIT = 100)
{
    // Initial Setup
    real tol {};
    real xh{}, xl{};
    real fh{}, fl{};
    real f{}, df{}, dfl{}, dfh{};
    real dx{}, dx_old{};
    real rts{};
    real temp{};
    std::tie(fl, dfl) = func(x1);
    std::tie(fh, dfh) = func(x2);

    // Check Provided [x1, x2] bracket
    if ((fl > 0.0 && fh > 0.0) || (fl < 0.0 && fh < 0.0))
        throw InvalidArgument("rtsafe: Root must be bracketed by x1 and x2");
    if (fl == 0.0) return RootResult { 0, true, x1, fl, dfl };
    if (fh == 0.0) return RootResult { 0, true, x2, fh, dfh };

    // Flip Bracket Depending on Sign
    if (fl < 0.0) 
    {
        xl = x1;
        xh = x2;
    } else
    {
        xl = x2;
        xh = x1;
    }

    // Initialize root guess, stepsize before last, and last step
    rts = 0.5 * (x1 + x2);
    dx_old = std::fabs(x2 - x1);
    dx = dx_old;
    std::tie(f, df) = func(rts);
    if (f == 0.0)
    { // Rare Early return where rts hits the root before the loop
        return RootResult { 0, true, rts, f, df };
    }
    for (idx j {0}; j < MAXIT; j++)
    {
        // Check condition for Bisection Step
        // Improvement on NR implementation is the finite check on f/df - forcing bisection
        if (!std::isfinite(f / df) || (((rts - xh) * df - f) * ((rts - xl) * df - f) > 0.0) 
            || (std::fabs(2.0 * f) > std::fabs(dx_old * df)))
        {
            dx_old = dx;
            dx = 0.5 * (xh - xl);
            rts = xl + dx;
            if (xl == rts)
            {
                std::tie(f, df) = func(rts);
                return RootResult { j, true, rts, f, df };
            }
        } else
        {
            dx_old = dx;
            dx = f / df;
            temp = rts;
            rts -= dx;
            if (temp == rts)
            {
                std::tie(f, df) = func(rts);
                return RootResult { j, true, rts, f, df };
            }
        }

        // Update Function
        std::tie(f, df) = func(rts);

        // Convergence Criteria
        tol = rtsafe_tol(rts, xacc);
        if (std::fabs(dx) < tol)
        {
            return RootResult { j, true, rts, f, df };
        }

        // Maintain Bracket
        if (f < 0.0)
            xl = rts;
        else
            xh = rts;
    }
    return RootResult { MAXIT, false, rts, f, df };
}

}

