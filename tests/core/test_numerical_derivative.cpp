#include <cstddef>
#include <climits>
#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <limits>
#include <span>
#include <format>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <stator/core/numerical_derivative.h>

using namespace stator::core;


namespace {

// COMPARISON TOLERANCE
real compare_tol(const real f, const real x, const real fac = 100.0)
{
    // Computes correct tolerance for comparing forward different derivatives
    // with analytical
    const real X { std::max(std::abs(x), 1.0) };
    return fac*std::sqrt(real_EPS)*(std::abs(f) / X + 1.0);
}

// Test Models (compliant with ValArgs Concept)
auto linear_model = [](real x, std::span<const real> a) -> real {
    return a[0] + a[1]*x;   
};

auto exponential_model = [](real x, std::span<const real> a) -> real {
    return a[0] * std::exp(-a[1]*x) + a[2];   
};

auto quadratic_model = [](real x, std::span<const real> a) -> real {
    return a[0] + a[1]*x + a[2]*x*x;   
};

auto cubic_model = [](real x, std::span<const real> a) -> real {
    return a[0] + a[1]*x + a[2]*x*x + a[3]*x*x*x;   
};

}

//==============================================================================
// INVALID ARGUMENT
//==============================================================================

//==============================================================================
// FUNCTION TESTS
//==============================================================================
TEST(Numerical_Derivative, Linear_Analytical)
{
    // Linear Function
    real x { 1.0 };
    NumericalDerivative nd { linear_model };
    std::vector<real> a { 1.0, 2.0 };

    const real f_expected { linear_model(x, a) };
    const real df_dx_expected { a[1] };

    // Result
    const std::pair<real, real> res { nd(x, a) };

    // Test
    EXPECT_DOUBLE_EQ(f_expected, res.first);
    EXPECT_NEAR(df_dx_expected, res.second, compare_tol(f_expected, x));
}

TEST(Numerical_Derivative, Exponential_Analytical)
{
    // Exponential Function: f(x, a) = a[0] * exp(-a[1]*x) + a[2]
    //                    dfdx(x, a) = -a[0]*a[1]*exp(-a[1]*x)
    real x { 69.420 };
    NumericalDerivative nd { exponential_model };
    std::vector<real> a { 1.0, 2.0, 3.0 };

    const real f_expected { exponential_model(x, a) };
    const real df_dx_expected { -a[0]*a[1]*std::exp(-a[1]*x) };

    // Result
    const std::pair<real, real> res { nd(x, a) };

    // Test
    EXPECT_DOUBLE_EQ(f_expected, res.first);
    EXPECT_NEAR(df_dx_expected, res.second, compare_tol(f_expected, x));
}

TEST(Numerical_Derivative, Exponential_Analytical_Disparate_Scales)
{
    // Exponential Function: f(x, a) = a[0] * exp(-a[1]*x) + a[2]
    //                    dfdx(x, a) = -a[0]*a[1]*exp(-a[1]*x)
    real x { 69.420 };
    NumericalDerivative nd { exponential_model };
    std::vector<real> a { 1e-6, 1e6, 3.0 };

    const real f_expected { exponential_model(x, a) };
    const real df_dx_expected { -a[0]*a[1]*std::exp(-a[1]*x) };

    // Result
    const std::pair<real, real> res { nd(x, a) };

    // Test
    EXPECT_DOUBLE_EQ(f_expected, res.first);
    EXPECT_NEAR(df_dx_expected, res.second, compare_tol(f_expected, x));
}

TEST(Numerical_Derivative, Quadratic_Analytical)
{
    // Quadratic Function x^2: Exercises f(x) = 0 performance
    real x { 0.0 };
    NumericalDerivative nd { quadratic_model };
    std::vector<real> a { 0.0, 0.0, 1.0 };

    const real f_expected { linear_model(x, a) };
    const real df_dx_expected { 2*a[2]*x };

    // Result
    const std::pair<real, real> res { nd(x, a) };

    // Test
    EXPECT_DOUBLE_EQ(f_expected, res.first);
    EXPECT_NEAR(df_dx_expected, res.second, compare_tol(f_expected, x));
}

TEST(Numerical_Derivative, Cubic_Analytical)
{
    // Quadratic Function x^2: Exercises f(x) = 0 performance
    real x { 0.0 };
    NumericalDerivative nd { cubic_model };
    std::vector<real> a { 0.0, 0.0, 0.0, 1.0 };

    const real f_expected { cubic_model(x, a) };
    const real df_dx_expected { 3*a[3]*x*x };

    // Result
    const std::pair<real, real> res { nd(x, a) };

    // Test
    EXPECT_DOUBLE_EQ(f_expected, res.first);
    EXPECT_NEAR(df_dx_expected, res.second, compare_tol(f_expected, x));
}