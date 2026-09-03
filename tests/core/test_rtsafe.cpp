#include <cstddef>
#include <climits>
#include <iostream>
#include <vector>
#include <utility>
#include <cmath>

#include <gtest/gtest.h>

#include <stator/core/rtsafe.h>

using namespace stator::core;

//==============================================================================
// INVALID ARGUMENT
//==============================================================================
TEST(Rtsafe, Unbracketed_Root)
{
    auto xsquared = [](real X, [[maybe_unused]] Params a) -> std::pair<real, real> {
        return { 
            X*X - 2, // f(x)
            2*X      // df(x)
         };
    };
    ParamVec a{};
    EXPECT_THROW((rtsafe(xsquared, a, 100.0, 105.0)), InvalidArgument);
}

//==============================================================================
// EARLY RETURN
//==============================================================================
TEST(Rtsafe, Root_at_Left_Endpoint)
{
    auto twox = [](real X, [[maybe_unused]] Params a) -> std::pair<real, real> {
        return { 
            2*X, // f(x)
            2.0  // df(x)
         };
    };
    ParamVec a{};
    RootResult res { rtsafe(twox, a, 0.0, 1.0) };
    EXPECT_TRUE(res.converged);
    EXPECT_EQ(res.iterations, 0);
    EXPECT_NEAR(res.x, 0.0, real_EPS);
    EXPECT_NEAR(res.f, 0.0, real_EPS);
    EXPECT_NEAR(res.df, 2.0, real_EPS); // Special Case (df is not stored by rtsafe)
}

TEST(Rtsafe, Root_at_Right_Endpoint)
{
    auto twox = [](real X, [[maybe_unused]] Params a) -> std::pair<real, real> {
        return { 
            2*X, // f(x)
            2.0  // df(x)
         };
    };
    ParamVec a{};
    RootResult res { rtsafe(twox, a, -1.0, 0.0) };
    EXPECT_TRUE(res.converged);
    EXPECT_EQ(res.iterations, 0);
    EXPECT_NEAR(res.x, 0.0, real_EPS);
    EXPECT_NEAR(res.f, 0.0, real_EPS);
    EXPECT_NEAR(res.df, 2.0, real_EPS); // Special Case (df is not stored by rtsafe)
}

//==============================================================================
// ROOT SWEEP
//==============================================================================
TEST(Rtsafe, Root_Sweep)
{
    // Tests Root Finds on Functions Where the Root Is Prescribed
    // f(x) = (x - r)*g(x)
    real x1 {};
    real x2 {};

    // Bracket Lambda
    auto brackets = [] (real r) -> std::pair<real, real> {
        const real h = std::max(std::fabs(r) * real{0.5}, real{1.0});
        return {r - 0.3 * h, r + 0.7 * h};
    };

    // Root Sweep
    ParamVec a{};
    std::vector<real> r { 0.0, 1.0, 1e8, 1e-8 };    
    for (idx i {}; i < r.size(); ++i)
    {
        std::tie(x1, x2) = brackets(r[i]);
        real r_val { r[i] };
        RootResult res {
            rtsafe(
                [r_val](real X, [[maybe_unused]] Params a) -> std::pair<real, real> {
                    return {
                        (X - r_val) * (X*X + 2.0),        // (x - r)*g(x)
                        (X*X + 2.0) + (X - r_val)*2.0*X   // g(x) + (x-r)*g'(x)
                    };
                },
                a, x1, x2, 0.0
            )
        };
        // Convergence & Tolerance Check
        EXPECT_TRUE(res.converged);
        EXPECT_LE(std::fabs(res.x - r_val), rtsafe_tol(res.x, 0.0));
    }
}

//==============================================================================
// MAX ITER (NO ROOT)
//==============================================================================
TEST(Rtsafe, Max_Iterations)
{
    idx MAXITER { 100 };
    auto noroot = [] (real X, [[maybe_unused]] Params a) -> std::pair<real, real> {
        return { X < 0.0 ? -1.0 : 1.0, 0.0 };
    };
    ParamVec a{};
    RootResult res {
        rtsafe(noroot, a, -1.0, 1.0, 0.0, MAXITER)
    };
    EXPECT_FALSE(res.converged);
    EXPECT_EQ(res.iterations, MAXITER);
}

//==============================================================================
// ADVERSARIAL FUNCTIONS
//==============================================================================
TEST(Rtsafe, Adversarial_Functions)
{
    // A suite of adversarial functions design to challenge rtsafe
    RootResult res {};
    ParamVec a{};

    // atan(1e6*x) - Bisection Must Save the Day
    res = rtsafe(
        [](real X, [[maybe_unused]] Params a) -> std::pair<real, real> {
            return {
                std::atan(1.0e6*X),
                1.0e6 / (1 + std::pow(1.0e6*X, 2))
            };
        }, a, -1.0, 1.0
    );
    EXPECT_TRUE(res.converged) << "Rtsafe failed on atan(1e6*x) - x = " << res.x << " - " << res.iterations << " iterations";

    // (x-1)^3 - Zero derivative root + divide by zero
    res = rtsafe(
        [](real X, [[maybe_unused]] Params a) -> std::pair<real, real> {
            return {
                std::pow(X-1, 3),
                3.0*std::pow(X-1, 2)
            };
        }, a, 0.0, 2.0, 0.0, 100
    );
    EXPECT_TRUE(res.converged) << "Rtsafe failed on (x-1)^3 - x = " << res.x << " - " << res.iterations << " iterations";

    // x^3 - 3x^2 + 3x - 1 - Noisy root
    res = rtsafe(
        [](real X, [[maybe_unused]] Params a) -> std::pair<real, real> {
            return {
                std::pow(X, 3) - 3.0*std::pow(X, 2) + 3.0*X - 1.0,
                3.0*std::pow(X, 2) - 6.0*X + 3.0
            };
        }, a, 0.0, 2.0, 0.0, 100
    );
    EXPECT_TRUE(res.converged) << "Rtsafe failed on x^3 - 3x^2 + 3x - x = " << res.x << " - " << res.iterations << " iterations";
}

//==============================================================================
// CONVERGENCE RATE
//==============================================================================
TEST(Rtsafe, Convergence_Rate)
{
    // x^2 - 1 - Smooth function should converge < 8 iterations
    ParamVec a{};
    RootResult res { 
        rtsafe(
            [](real X, [[maybe_unused]] Params a) -> std::pair<real, real> {
                return {
                    std::pow(X, 2) - 1.0,
                    2.0*X
                };
            }, a, 0.0, 2.0
        )
    };
    EXPECT_LE(res.iterations, 8) << "Rtsafe not converging fast enough for smooth function";
}

TEST(Rtsafe, Numerical_Derivative)
{
    // Test Rtsafe against an adversarial function, compared with numerical derivative overload version
    // (x-1)^3 - Zero derivative root + divide by zero
    ParamVec a{};
    RootResult res_an = rtsafe(
        [](real X, [[maybe_unused]] Params a) -> std::pair<real, real> {
            return {
                std::pow(X-1, 3),
                3.0*std::pow(X-1, 2)
            };
        }, a, 0.0, 2.0, 0.0, 100
    );
    RootResult res_num = rtsafe(
        [](real X, [[maybe_unused]] Params a) -> real {
            return std::pow(X-1, 3);
        }, a, 0.0, 2.0, 0.0, 100
    );

    // Both Should Converge
    EXPECT_TRUE(res_an.converged) << "Rtsafe failed on (x-1)^3 - x = " << res_an.x << " - " << res_an.iterations << " iterations";
    EXPECT_TRUE(res_num.converged) << "Rtsafe failed on (x-1)^3 - x = " << res_num.x << " - " << res_num.iterations << " iterations";

    // Compare Root
    EXPECT_NEAR(res_an.x, res_num.x, 1e-10) << "Rtsafe with numerical derivative differs from analytical";
}