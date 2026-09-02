#pragma once

#include <cmath>
#include <vector>

#include <stator/core/constants.h>
#include <stator/core/concepts.h>
#include <stator/core/types.h>
#include <stator/core/errors.h>

namespace stator::core {

// A callable representing the 
template <Model F>
class NumericalJacobian {
    F m_func;
    mutable std::vector<real> m_a_pda;   // a + da
public:
    NumericalJacobian(F func, idx ma)
        : m_func { std::move(func) }
        , m_a_pda(ma, 0.0)
    {}

    // Based on NRfdjac (Numerical Recipes, 3rd Edition)
    // Numeric derivate based on forward difference
    real operator()(real x, const std::vector<real>& a, std::span<real> dyda) const
    {
        using std::abs;

        // Argument Check
        if (a.size() != dyda.size())
            throw InvalidArgument("NumericalJacobian(): param vector a does not match dyda vector in length.");
        if (a.size() != m_a_pda.size())
            throw InvalidArgument("NumericalJacobian(): param vector a does not match size passed via constructor.");

        // Setup
        // Copy a into a_pda
        std::copy(a.begin(), a.end(), m_a_pda.begin());
        const real EPS { std::sqrt(real_EPS) };
        real f_pdf {};

        // Function Call
        real f { m_func(x, a) };

        for (idx j {}; j < a.size(); ++j)
        {
            // Bookkeeping and Step Calculation
            real temp { m_a_pda[j] };
            real h { EPS*abs(temp) };
            if (h == 0.0 ) h = EPS;
            m_a_pda[j] = temp + h;
            h = m_a_pda[j] - temp;

            // Evaluate f(a_pda)
            f_pdf = m_func(x, m_a_pda);

            // Partial Derivative
            dyda[j] = (f_pdf - f) / h;

            // Restore a_pda
            m_a_pda[j] = temp;
        }

        // Clean temporary state
        std::fill(m_a_pda.begin(), m_a_pda.end(), 0.0);

        return f;
    }
};

}
