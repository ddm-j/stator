#pragma once

#include <cmath>
#include <utility>
#include <span>
#include <vector>
#include <cassert>

#include <linalg/Matrix.h>
#include <linalg/solvers.h>

#include <stator/core/constants.h>
#include <stator/core/concepts.h>
#include <stator/core/errors.h>
#include <stator/core/types.h>
#include <stator/core/numeric_result.h>
#include <stator/core/numerical_jacobian.h>

namespace stator::core {

namespace detail {

inline real median(std::vector<real>& x)
{
    if (x.empty())
        return 0.0;
    
    idx n = x.size();
    auto middle = x.begin() + static_cast<std::ptrdiff_t>(n) / 2;
    std::nth_element(x.begin(), middle, x.end());
    if (n % 2 != 0) {
        // Odd number of elements: return the exact middle element
        return *middle;
    } else {
        // Even number of elements: average of middle and the largest element before it
        auto left_middle = std::max_element(x.begin(), middle);
        return (*middle + *left_middle) / 2.0;
    }
}

inline real rofunc(const std::vector<real>& x, const std::vector<real>& y, std::vector<real>& buf, real& a, const real b, real& abdev)
{
    // Asserts
    assert(x.size() == y.size());
    assert(y.size() == buf.size());

    // Setup
    idx j {};
    real d {}, sum {};

    // Fill Buffer with Residuals
    std::transform(x.begin(), x.end(), y.begin(), buf.begin(), [b](auto x, auto y) { return y - b * x; });

    // Set A = median residual
    a = median(buf);
    abdev = 0.0;
    for (j = 0; j < buf.size(); j++)
    {
        d = y[j] - (b * x[j] + a);
        abdev += std::abs(d);
        if (y[j] != 0.0) d /= std::abs(y[j]);
        if (std::abs(d) > real_EPS) sum += (d >= 0.0 ? x[j] : -x[j]);
    }
    return sum;
}

}

// Improved Convergence implementation of NR edition 3's fitmed
// Iterates until slope changes are at machine precision
inline LinRegResult fitmed(Params x, Params y)
{
    if (x.size() != y.size())
        throw InvalidArgument("fitmed(): x and y don't have matching sizes");

    // Setup
    real a {}, b {}, abdev {};
    idx j {};
    idx ndata { x.size() };
    std::vector<real> buf(x.size(), 0.0);
    real b1 {}, b2{}, del {}, f {}, f1 {}, f2 {}, sigb {}, temp {};
    real sx {}, sy {}, sxy {}, sxx {}, chisq {};

    // Initialize a and b with least squares guess
    for (j = 0; j < ndata; j++)
    {
        sx += x[j];
        sy += y[j];
        sxy += x[j]*y[j];
        sxx += x[j]*x[j];
    }
    del = static_cast<real>(ndata) * sxx - sx * sx;
    a = (sxx*sy - sx*sxy) / del;
    b = (static_cast<real>(ndata) * sxy - sx * sy) / del;

    for (j = 0; j < ndata; j++)
        chisq += (temp = y[j] - (a + b * x[j]), temp*temp);
    sigb = std::sqrt(chisq / del);

    b1 = b;
    f1 = detail::rofunc(x, y, buf, a, b1, abdev);
    if (sigb > 0.0)
    {
        b2 = b + std::copysign(3.0*sigb, f1);
        f2 = detail::rofunc(x, y, buf, a, b2, abdev);
        if (b2 == b1) {
            // Degenerate Bracket
            abdev /= static_cast<real>(ndata);
            LinRegResult res { a, b, abdev };
            return res;
        }

        while (f1*f2 > 0.0)
        {
            b = b2 + 1.6 * (b2 - b1);
            b1 = b2;
            f1 = f2;
            b2 = b;
            f2 = detail::rofunc(x, y, buf, a, b2, abdev);
        }
        // sigb *= 0.01;
        while (std::abs(b2 - b1) > 2 * real_EPS*(std::max(std::abs(b2), std::abs(b1))))
        {
            b = b1 + 0.5 * (b2 - b1);
            if ((b == b1) || (b == b2)) break;
            f = detail::rofunc(x, y, buf, a, b, abdev);
            if (f * f1 >= 0.0) {
                f1 = f;
                b1 = b;
            } else {
                f2 = f;
                b2 = b;
            }
        }
    }
    abdev /= static_cast<real>(ndata);
    LinRegResult res { a, b, abdev };
    return res;
}

}