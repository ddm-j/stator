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

#include <stator/core/numerical_jacobian.h>

using namespace stator::core;

// COMPARISON TOLERANCE
real compare_tol(const real f, const real ai, const real fac = 100.0)
{
    // Computes correct tolerance for comparing forward different derivatives
    // with analytical
    real f_abs { std::fabs(f) };
    return fac*std::sqrt(real_EPS)*f_abs / std::max(ai, 1.0);
}

// Test Models (compliant with ValArgs Concept)
auto linear_model = [](real x, std::span<const real> a) -> real {
    return a[0] + a[1]*x;   
};
// static_assert(ArgsVal<decltype(linear_model)>);

auto exponential_model = [](real x, std::span<const real> a) -> real {
    return a[0] * std::exp(-a[1]*x) + a[2];   
};
// static_assert(ArgsVal<decltype(exponential_model)>);

//==============================================================================
// INVALID ARGUMENT
//==============================================================================
TEST(Numerical_Jacobian, Wrong_Shape_To_Constructor)
{
    NumericalJacobian nj { linear_model, 3 }; // Deliberately wrong size
    std::vector<real> a { 1.0, 2.0 };
    std::vector<real> dyda { 0.0, 2.0 };
    EXPECT_THROW(nj(0.0, a, dyda), InvalidArgument);
}

TEST(Numerical_Jacobian, Wrong_Shape_a_Mismatch)
{
    NumericalJacobian nj { linear_model, 2 };
    std::vector<real> a { 1.0, 2.0, 3.0 };
    std::vector<real> dyda { 0.0, 2.0 };
    EXPECT_THROW(nj(0.0, a, dyda), InvalidArgument);
}

//==============================================================================
// FUNCTION TESTS
//==============================================================================
TEST(Numerical_Jacobian, Linear_Analytical)
{
    // Linear Function
    real x { 1.0 };
    NumericalJacobian nj { linear_model, 2 };
    std::vector<real> a { 1.0, 2.0 };
    std::vector<real> dyda { 0.0, 2.0 };
    real f { nj(x, a, dyda) };

    std::vector<real> expected { 1.0, x };
    for (idx i {}; i < a.size(); ++i)
    {
        EXPECT_NEAR(dyda[i], expected[i], compare_tol(f, a[i]));
    }
}

TEST(Numerical_Jacobian, Exponential_Analytical)
{
    // Exponential Function: f(x, a) = a[0] * exp(-a[1]*x) + a[2]
    real x { 69.420 };
    NumericalJacobian nj { exponential_model, 3 };
    std::vector<real> a { 1.0, 2.0, 3.0 };
    std::vector<real> dyda { 0.0, 0.0, 0.0 };
    real f { nj(x, a, dyda) };

    std::vector<real> expected { std::exp(-a[1]*x),  -a[0]*a[1]*std::exp(-a[1]*x), 1.0 };
    for (idx i {}; i < a.size(); ++i)
    {
        EXPECT_NEAR(dyda[i], expected[i], compare_tol(f, a[i]));
    }
}

TEST(Numerical_Jacobian, Exponential_Analytical_Disparate_Scales)
{
    // Exponential Function: f(x, a) = a[0] * exp(-a[1]*x) + a[2] (with diparate params)
    real x { 3.14 };
    NumericalJacobian nj { exponential_model, 3 };
    std::vector<real> a { 1e-6, 1e6, 2.0 };
    std::vector<real> dyda { 0.0, 0.0, 0.0 };
    real f { nj(x, a, dyda) };

    std::vector<real> expected { std::exp(-a[1]*x),  -a[0]*a[1]*std::exp(-a[1]*x), 1.0 };
    for (idx i {}; i < a.size(); ++i)
    {
        EXPECT_NEAR(dyda[i], expected[i], compare_tol(f, a[i]));
    }
}

TEST(Numerical_Jacobian, Exponential_Zero_Parameter)
{
    // Exponential Function: f(x, a) = a[0] * exp(-a[1]*x) + a[2] (a[0] = 0.0)
    real x { 3.14 };
    NumericalJacobian nj { exponential_model, 3 };
    std::vector<real> a { 0.0, 10.0, 1.0 };
    std::vector<real> dyda { 0.0, 0.0, 0.0 };
    real f { nj(x, a, dyda) };

    std::vector<real> expected { std::exp(-a[1]*x), 0.0, 1.0 };
    for (idx i {}; i < a.size(); ++i)
    {
        EXPECT_NEAR(dyda[i], expected[i], compare_tol(f, a[i]));
    }
}

TEST(Numerical_Jacobian, Repeated_Calls)
{
    // NumericalJacobian object is stateful. Ensure that it returns same results after repeated calls

    // Linear Function
    real x { 1.0 };
    NumericalJacobian nj { linear_model, 2 };
    std::vector<real> a { 1.0, 2.0 };
    std::vector<real> dyda1 { 0.0, 2.0 };
    [[maybe_unused]] real f { nj(x, a, dyda1) };

    std::vector<real> dyda2 { 0.0, 2.0 };
    f = nj(x, a, dyda2);

    for (idx i {}; i < a.size(); ++i)
    {
        EXPECT_EQ(dyda1[i], dyda2[i]);
    }
}