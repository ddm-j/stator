#include <cstddef>
#include <climits>
#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <limits>
#include <format>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <stator/core/golden_section.h>

using namespace stator::core;

// Bracket Lambda - Creates [a, b] pairs for an arbitrary root point
auto create_brackets = [] (real r) -> std::pair<real, real> {
    const real h = std::max(std::fabs(r) * real {0.5}, real {1.0});
    return {r - 0.3 * h, r + 0.7 * h};
};

//==============================================================================
// INVALID ARGUMENT
//==============================================================================
TEST(Golden_Section, Bad_Tol_Throws)
{
    auto xsquared = [](real X, [[maybe_unused]] Params a) -> real {
        return std::pow(X, 2);   
    };
    ParamVec a{};
    EXPECT_THROW(golden_section(xsquared, a, -1.0, 1.0, -1e-8), InvalidArgument);
    EXPECT_THROW(golden_section(xsquared, a, -1.0, 1.0, std::numeric_limits<real>::quiet_NaN()), InvalidArgument);
}

//==============================================================================
// MINIMUM SWEEP
//==============================================================================
TEST(Golden_Section, Minimum_Sweep)
{
    // Tests Minimizer Against prescribed minimums
    // f(x) = (x - m^2)
    real x1 {};
    real x2 {};
    const real tol { std::sqrt(real_EPS) };

    // Minimum Sweep
    ParamVec a{};
    std::vector<real> m { 0.0, -0.1, 0.1, -100.0, 100.0, -200.0, 200.0 };    
    for (idx i {}; i < m.size(); ++i)
    {
        // Setup Minimization Problem
        std::tie(x1, x2) = create_brackets(m[i]);
        real m_val { m[i] };
        auto func = [m_val](real X, [[maybe_unused]] Params a) -> real {
            return std::pow(X - m_val, 2);
        };
        Bracket brack { bracket(func, a, x1, x2) };
        MinResult1D res { golden_section(func, a, brack, tol)};

        // Convergence & Tolerance Check
        // Is converged?
        EXPECT_TRUE(res.converged);
        // Theoretical Convergence Rate?
        EXPECT_LE(res.iterations, static_cast<idx>(std::log(tol)/std::log(GOLD_R))+2);
        // Tolerance correct?
        EXPECT_LE(std::fabs(res.x - m_val), tol*(2*std::fabs(m_val) + brack.get_width()));
    }
}

//==============================================================================
// MAX ITERATIONS
//==============================================================================
TEST(Golden_Section, Maximum_Iterations)
{
    const real tol { std::sqrt(real_EPS) };
    const idx MAXIT { 5 };
    auto xsquared = [](real X, [[maybe_unused]] Params a) -> real {
        return std::pow(X, 2);   
    };
    ParamVec a{};

    real x1 {}, x2 {};
    std::tie(x1, x2) = create_brackets(10.0);
    Bracket brack { bracket(xsquared, a, x1, x2) };
    MinResult1D res { golden_section(xsquared, a, brack, tol, MAXIT) };

    // Failed to Converge?
    EXPECT_FALSE(res.converged);
    // Maximum Iteration Branch?
    EXPECT_EQ(res.iterations, MAXIT);
    // Tolerance Check
    EXPECT_GT(std::fabs(res.x - 0.0), tol*(2*std::fabs(0.0) + brack.get_width()));

}