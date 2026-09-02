#include <cstddef>
#include <climits>
#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <stator/core/golden_section.h>

using namespace stator::core;

//==============================================================================
// INVALID ARGUMENT
//==============================================================================
TEST(Bracket, Equal_Brackets_Throws)
{
    auto xsquared = [](real X, Params a) -> real {
        return a[0] + a[1]*X + a[2]*std::pow(X, 2);   
    };
    ParamVec a { 0, 0, 1.0 };
    EXPECT_THROW(bracket(xsquared, a, 1.0, 1.0), InvalidArgument);
}

TEST(Bracket, Equal_Fa_Fb_Throws)
{
    auto xsquared = [](real X, Params a) -> real {
        return a[0] + a[1]*X + a[2]*std::pow(X, 2);   
    };
    ParamVec a { 0, 0, 1.0 };
    EXPECT_THROW(bracket(xsquared, a, 1.0, -1.0), InvalidArgument);
}

TEST(Bracket, NonFinite_Bracket_Throws)
{
    auto xsquared = [](real X, Params a) -> real {
        return a[0] + a[1]*X + a[2]*std::pow(X, 2);   
    };
    ParamVec a { 0, 0, 1.0 };
    // Infinity
    EXPECT_THROW(bracket(xsquared, a, std::numeric_limits<real>::infinity(), 1.0), InvalidArgument);
    EXPECT_THROW(bracket(xsquared, a, 1.0, std::numeric_limits<real>::infinity()), InvalidArgument);

    // NaN
    EXPECT_THROW(bracket(xsquared, a, std::numeric_limits<real>::quiet_NaN(), 1.0), InvalidArgument);
    EXPECT_THROW(bracket(xsquared, a, 1.0, std::numeric_limits<real>::quiet_NaN()), InvalidArgument);
}

TEST(Bracket, NonFinite_Before_Loop_Throws)
{
    real NaNat {};
    // Returns NaN when X = NaNat otherwise f(X) = X
    auto returnsNaN = [&NaNat](real X, [[maybe_unused]] Params a) -> real {
        if (X == NaNat)
            return std::numeric_limits<real>::quiet_NaN();
        else
            return X;
    };

    ParamVec a{};
    // f(a) = NaN
    NaNat = 1.0;
    EXPECT_THROW(bracket(returnsNaN, a, NaNat, 2.0), InvalidArgument);

    // f(b) = NaN
    NaNat = 2.0;
    EXPECT_THROW(bracket(returnsNaN, a, 1.0, NaNat), InvalidArgument);
}

TEST(Bracket, NonFinite_In_Loop_Throws)
{
    real alim { -5.0 };
    real blim { 5.0 };
    idx call_counter { 0 };
    // Returns NaN when X not in [alim, blim] else f(X) = X
    auto returnsNaN_ranged = [alim, blim, &call_counter](real X, [[maybe_unused]] Params a) -> real {
        ++call_counter;
        if ((X < alim) || (X > blim))
            return std::numeric_limits<real>::quiet_NaN();
        else
            return X;
    };
    ParamVec a{};

    // Bracket expands until a NaN is created
    using testing::HasSubstr;
    using testing::Not;
    try {
        bracket(returnsNaN_ranged, a, 1.0, 2.0);
        FAIL() << "expected InvalidArgument";
    } catch (const InvalidArgument& e) {
        const std::string msg = e.what();
        EXPECT_THAT(msg, Not(HasSubstr("iteration 0")));
    }
}

//==============================================================================
// CONVERGENCE FAILURE
//==============================================================================
TEST(Bracket, Max_Iterations_Throws)
{
    // Linear function is monotonic
    // Breaks calling contract (no minimum)
    auto neg_X = [](real X, [[maybe_unused]] Params a) -> real {
        return -X;
    };
    ParamVec a{};

    // We're passing a low iteration count to avoid overflow 
    EXPECT_THROW(bracket(neg_X, a, 0.0, 1.0, 10), ConvergenceFailure);
}

//==============================================================================
// FUNCTION SUITES
//==============================================================================
TEST(Bracket, Function_Branch_Sweep)
{
    // Should Exercise All Code Branches and Return Valid Brackets
    real a {0.0}, b {1.0};
    Bracket res {};

    // X^2 Family
    // Early Exit
    auto early_exit = [](real X, [[maybe_unused]] Params a) -> real {
        return std::pow(X - 1.5, 2);
    };
    ParamVec p{};

    res = bracket(early_exit, p, a, b);
    ASSERT_TRUE(res.is_valid());

    // A.1 Branch
    auto A_1_branch = [](real X, [[maybe_unused]] Params a) -> real {
        return std::pow(X - 2.2, 2);
    };
    res = bracket(A_1_branch, p, a, b);
    ASSERT_TRUE(res.is_valid());

    // A.2 Branch
    auto A_2_branch = [](real X, [[maybe_unused]] Params a) -> real {
        return std::pow(X-3.0, 2) + 10.0*std::exp(-std::pow((X-2.2)/0.3, 2));
    };
    res = bracket(A_2_branch, p, a, b);
    ASSERT_TRUE(res.is_valid());

    // B Branch
    auto B_branch = [](real X, [[maybe_unused]] Params a) -> real {
        return std::pow(X-10.0, 2);
    };
    res = bracket(B_branch, p, a, b);
    ASSERT_TRUE(res.is_valid());

    // C Branch
    auto C_branch = [](real X, [[maybe_unused]] Params a) -> real {
        return std::pow(X-200.0, 2);
    };
    res = bracket(C_branch, p, a, b);
    ASSERT_TRUE(res.is_valid());

    // D Branch
    auto D_branch = [](real X, [[maybe_unused]] Params a) -> real {
        return -std::exp(-std::pow(X-20.0, 2));
    };
    res = bracket(D_branch, p, a, b);
    ASSERT_TRUE(res.is_valid());

    // BONUS: Plateau Underflow Throw
    auto plateau = [](real X, [[maybe_unused]] Params a) -> real {
        return -std::tanh(X);
    };
    EXPECT_THROW(bracket(plateau, p, a, b), ConvergenceFailure);
    
}