#pragma once

#include <cmath>
#include <vector>

#include <stator/core/constants.h>
#include <stator/core/concepts.h>
#include <stator/core/types.h>
#include <stator/core/errors.h>

namespace stator::core {

// A callable complying with DifferentiableModel concept
// with only an analytical Model (uses forward difference derivative) 
template <Model F>
class NumericalDerivative {
    F m_func;
public:
    NumericalDerivative(F func)
        : m_func { std::move(func) }
    {}

    // Simple forward difference derivative
    std::pair<real, real> operator()(real x, const std::vector<real>& a) const
    {
        using std::abs;

        // Argument Check

        // Setup
        const real EPS { std::sqrt(real_EPS) };
        real f_pdf {};
        real temp { x };
        real h {};
        real x_pdx {};
        real df_dx {};

        // Function Call
        real f { m_func(x, a) };

        // Preturb
        h = EPS*abs(temp);
        if (h == 0.0) h = EPS;
        x_pdx = temp + h;
        h = x_pdx - temp;        // floating point trick from NR 3rd edition
        f_pdf = m_func(x_pdx, a);

        // Derivative
        df_dx = (f_pdf - f) / h;

        return {f, df_dx};
    }
};

}