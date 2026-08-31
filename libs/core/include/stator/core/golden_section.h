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

struct Bracket
{
    real ax {}, bx {}, cx {};
    real fa {}, fb {}, fc {};

    bool is_valid() const
    {
        auto [min, max] = std::minmax(ax, cx);
        return 
            (
                (max > bx) && (bx > min) // b is in [a, c]
            ) &&
            (
                (fa > fb) && (fc > fb) // b is a minimum
            );
    }

    real get_width() const
    {
        return std::fabs(cx - ax);
    }
};

// Implementation Details
namespace detail {
    inline void shft3(real &a, real &b, real &c, const real &d)
    {
        a = b;
        b = c;
        c = d;
    }

    inline void shft2(real &a, real &b, const real &c)
    {
        a = b;
        b = c;
    }

    // Safe function evalutation (ensure finite)
    template <ValOnly F>
    inline real eval(F&& func, const real x, const idx it)
    {
        const real f { func(x) };
        if (!std::isfinite(f))
            throw InvalidArgument("bracket: func({}) returned non-finite value {} at iteration {}", x, f, it);
        return f;
    }

}


// 1d minimum bracketing - Numerical Recipes, Third Edition (improved)
// Contract: Garaunteed to return a valid bracket for 1D minimization
//           throws otherwise.
//           func(x) is ALWAYS wrapped with eval(func, x, iteration), this
//           garauntees that we don't need to test every return branch. It is
//           impossible to return non-finite values.
template <ValOnly F>
Bracket bracket(F&& func, const real a, const real b, const idx MAXIT = 100)
{
    // Argument Check
    if (a == b)
    {
        throw InvalidArgument("bracket: [a, b] bracket = [{}, {}], a and b must be distinct.", a, b);
    }
    if (!std::isfinite(a) || !std::isfinite(b))
    {
        throw InvalidArgument("bracket: NaN or Inf in initial bracket [a, b]");
    }

    // Setup
    idx iter { 0 };
    real ax { a };
    real bx { b };
    real cx {};
    real fu {};
    real fa { detail::eval(func, ax, iter) };
    real fb { detail::eval(func, bx, iter) };
    if (fa == fb)
    {
        throw InvalidArgument("bracket: fa(a) == fa(b) = {}. bracket cannot determine a search direction.", fa);
    }
    real fc {};
    real r {}, q {}, u {};
    real ulim {};

    // Force Downhill Orientation for Bracketing
    if (fb > fa)
    {
        std::swap(ax, bx);
        std::swap(fb, fa);
    }

    // Start Bracket Loop
    cx = bx + GOLD * (bx - ax);
    fc = detail::eval(func, cx, iter);
    while (fb > fc)
    {
        // Maximum Iteration Check
        if (++iter > MAXIT)
        {
            // Maximum Iterations Achieved without a valid bracket
            throw ConvergenceFailure(
                "bracket: no valid bracket found after {} iterations (MAXIT). "
                "User supplied function appears monotonic on [{}, {}]. "
                "Monotonic functions cannot be minimized.", MAXIT, ax, cx
            );
        }

        // Quadratic Interpolation -> u = parabola vertex
        r = (bx - ax) * (fb - fc);
        q = (bx - cx) * (fb - fa);
        u = bx - ((bx - cx) * q - (bx - ax) * r) /
                (2.0 * std::copysign(std::max(std::fabs(q - r), TINY), q - r));
        ulim = bx + GLIMIT * (cx - bx);

        // Branch A: u between [b, c] 
        if ((bx - u) * (u - cx) > 0.0)
        {
            fu = detail::eval(func, u, iter);
            if (fu < fc) // A.1: Valid Bracket, Minimum Between b and c
            {
                ax = bx;
                bx = u;
                fa = fb;
                fb = fu;
                break;
            } else if (fu > fb) // A.2 Valid Bracket, Minimum Between a and u
            {
                cx = u;
                fc = fu;
                break;
            }
            // Quadratic Interpolation did nothing
            u = cx + GOLD * (cx - bx);
            fu = detail::eval(func, u, iter);
        } 
        // Branch B: u between [c, ulim]
        else if ((cx - u) * (u - ulim) > 0.0)
        {
            fu = detail::eval(func, u, iter);
            if (fu < fc)
            {
                detail::shft3(bx, cx, u, u + GOLD * (u - cx));
                detail::shft3(fb, fc, fu, detail::eval(func, u, iter));
            }
        }
        // Branch C: u limited to ulim
        else if ((u - ulim) * (ulim - cx) >= 0.0)
        {
            u = ulim;
            fu = detail::eval(func, u, iter);
        }
        // Branch D: Reject Parabolic Step, use default step instead
        else 
        {
            u = cx + GOLD * (cx - bx);
            fu = detail::eval(func, u, iter);
        }
        detail::shft3(ax, bx, cx, u);
        detail::shft3(fa, fb, fc, fu);

        // Debug Assertion - did you break something?
        assert(std::isfinite(cx) && std::isfinite(fc));
    }

    // Special Case Check (flat function)
    if (fb == fc)
    {
        throw ConvergenceFailure(
            "bracket: equal function values f({}) == f({}) == {}; cannot determine "
            "search direction from this interval.", bx, cx, fb
        );
    }

    Bracket result { ax, bx, cx, fa, fb, fc };
    assert(result.is_valid());
    return result;
}

// golden section search, 1d minimization routine - Numerical Recipes, Third Edition
// Contract: Unbounded Minimization
//           User provided bracket [a, b] undergoes expansion to satisfy that
//           the bracket contains a minimum before minimizatoin starts.
//           Convergence in n = log(tol)/log(0.618)
//           Tolerance is clamped to sqrt(eps) if caller specifies anything lower 
template <ValOnly F>
MinResult1D golden_section(F&& func, const Bracket brack, real tol = 0.0, const idx MAXIT = 100)
{
    // Argument Check
    assert(brack.is_valid());
    if ((tol < 0) || !std::isfinite(tol))
        throw InvalidArgument("golden_section: Invalid tolerance: {}. Must be finite greater than zero.", tol);

    // Tolerance Clamp
    if (tol < std::sqrt(real_EPS))
        tol = std::sqrt(real_EPS);

    // Setup
    const real W0 { brack.get_width() }; // Initial Bracket Width
    real x0 {}, x1 {}, x2 {}, x3 {};
    idx iter {};
    x0 = brack.ax;
    x3 = brack.cx;
    if (std::fabs(brack.cx - brack.bx) > std::fabs(brack.bx - brack.ax))
    {
        x1 = brack.bx;
        x2 = brack.bx + GOLD_C * (brack.cx - brack.bx);
    }
    else
    {
        x2 = brack.bx;
        x1 = brack.bx - GOLD_C * (brack.bx - brack.ax);
    }

    // Initial Function Evaluations
    real f1 { detail::eval(func, x1, iter) };
    real f2 { detail::eval(func, x2, iter) };

    // Loop
    while (std::fabs(x3 - x0) > tol*(std::fabs(x1) + std::fabs(x2) + W0))
    {
        // Iteration Guard
        if (++iter > MAXIT)
        {
            return MinResult1D { iter-1, false, x1, f1, std::fabs(x0 - x3) };
        }

        if (f2 < f1)
        {
            detail::shft3(x0, x1, x2, GOLD_R * x2 + GOLD_C * x3);
            detail::shft2(f1, f2, detail::eval(func, x2, iter));
        }
        else
        {
            detail::shft3(x3, x2, x1, GOLD_R * x1 + GOLD_C * x0);
            detail::shft2(f2, f1, detail::eval(func, x1, iter));
        }
    }
    if (f1 < f2)
        return MinResult1D { iter, true, x1, f1, std::fabs(x3 - x0) };
    else
        return MinResult1D { iter, true, x2, f2, std::fabs(x3 - x0) };
}

template <ValOnly F>
MinResult1D golden_section(F&& func, const real a, const real b, real tol = 0.0, const idx MAXIT = 100)
{
    return golden_section(func, bracket(func, a, b), tol, MAXIT);
}

}

