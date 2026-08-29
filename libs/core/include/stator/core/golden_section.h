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
};

// Implementation Details
namespace detail {
    // Shifts by reference right -> left
    // Avoids need of temp locals in bracket()
    // Note: d is const, because nothing goes into d, we can bind an rvalue to it
    inline void shft3(real &a, real &b, real &c, const real &d)
    {
        a = b;
        b = c;
        c = d;
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
    constexpr real GOLD { 1.618034 };
    constexpr real GLIMIT { 100.0 }; 
    constexpr real TINY { 1e-20 };
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

}

