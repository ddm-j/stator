#include <cstddef>
#include <climits>
#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <limits>
#include <format>
#include <string>

#include <linalg/Matrix.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <stator/core/fitmrq.h>
#include <data/nist_data.h>

using namespace stator::core;
using linalg::Matrix;

namespace {

// Test Models (compliant with ArgsValJac Concept)
auto linear_model_jac = [](real x, const std::vector<real>& a, std::vector<real>& dyda) -> real {
    dyda[0] = 1.0;
    dyda[1] = x;
    return a[0] + a[1]*x;   
};

auto quadratic_model_jac = [](real x, const std::vector<real>& a, std::vector<real>& dyda) -> real {
    dyda[0] = 1.0;
    dyda[1] = x;
    dyda[2] = x * x;
    return a[0] + a[1] * x + a[2] * x * x;   
};

auto non_polynomial_jac = [](real x, const std::vector<real>& a, std::vector<real>& dyda) -> real {
    dyda[0] = 1.0;
    dyda[1] = x;
    dyda[2] = 1.0 / (x * x);
    return a[0] + a[1] * x + a[2] / (x * x);
};

}

//==============================================================================
// INVALID ARGUMENT TESTS
//==============================================================================
struct FitmrqInvalidArgsCase {
    std::string name;              // for readable test IDs
    // Inputs
    real (*func) (real, const std::vector<real>&, std::vector<real>&);
    std::vector<real> x, y, sig;
    std::vector<real> a;
    std::vector<bool> ia;
    real tol;
    idx MAXIT;
    // Outputs
    // { no expected outputs, these are failure cases }
};
void PrintTo(const FitmrqInvalidArgsCase& c, std::ostream* os) { *os << c.name; }

class FitmrqTest : public ::testing::TestWithParam<FitmrqInvalidArgsCase> {};

TEST_P(FitmrqTest, Invalid_Argument)
{
    // Get Test Parameters
    const FitmrqInvalidArgsCase& c = GetParam();

    // Call mrqcof
    EXPECT_THROW(fitmrq(c.func, c.x, c.y, c.sig, c.a, c.ia, c.tol, c.MAXIT), InvalidArgument);
}

// Define Test Cases
INSTANTIATE_TEST_SUITE_P(
    InvalidArgument,
    FitmrqTest,
    ::testing::Values(
        FitmrqInvalidArgsCase{ "size_x_ne_y",
        linear_model_jac,
        std::vector<real>{ 0, 1, 2 },    // x
        std::vector<real>{ 1, 3, 5, 4 }, // y
        std::vector<real>{ 1, 1, 1 },    // sig
        std::vector<real>{ 0.0, 0.0 },   // a
        std::vector<bool>{ true, true }, // ia
        real { 0.0 },                    // tol
        idx { 1000 }                     // MAXIT
        },
        FitmrqInvalidArgsCase{ "size_y_ne_sig",
        linear_model_jac,
        std::vector<real>{ 0, 1, 2 },    // x
        std::vector<real>{ 1, 3, 5 }, // y
        std::vector<real>{ 1, 1, 1, 1 },    // sig
        std::vector<real>{ 0.0, 0.0 },   // a
        std::vector<bool>{ true, true }, // ia
        real { 0.0 },                    // tol
        idx { 1000 }                     // MAXIT
        },
        FitmrqInvalidArgsCase{ "size_a_ne_ia",
        linear_model_jac,
        std::vector<real>{ 0, 1, 2 },    // x
        std::vector<real>{ 1, 3, 5 }, // y
        std::vector<real>{ 1, 1, 1,},    // sig
        std::vector<real>{ 0.0, 0.0 },   // a
        std::vector<bool>{ true, true, true }, // ia
        real { 0.0 },                    // tol
        idx { 1000 }                     // MAXIT
        },
        FitmrqInvalidArgsCase{ "size_ia_all_disabled",
        linear_model_jac,
        std::vector<real>{ 0, 1, 2 },    // x
        std::vector<real>{ 1, 3, 5 }, // y
        std::vector<real>{ 1, 1, 1,},    // sig
        std::vector<real>{ 0.0, 0.0 },   // a
        std::vector<bool>{ false, false }, // ia
        real { 0.0 },                    // tol
        idx { 1000 }                     // MAXIT
        },
        FitmrqInvalidArgsCase{ "non_finite_x",
        linear_model_jac,
        std::vector<real>{ 0, 1, std::numeric_limits<real>::quiet_NaN() },    // x
        std::vector<real>{ 1, 3, 5 }, // y
        std::vector<real>{ 1, 1, 1,},    // sig
        std::vector<real>{ 0.0, 0.0 },   // a
        std::vector<bool>{ true, true }, // ia
        real { 0.0 },                    // tol
        idx { 1000 }                     // MAXIT
        },
        FitmrqInvalidArgsCase{ "non_finite_y",
        linear_model_jac,
        std::vector<real>{ 0, 1, 2 },    // x
        std::vector<real>{ 1, 3, std::numeric_limits<real>::quiet_NaN() }, // y
        std::vector<real>{ 1, 1, 1,},    // sig
        std::vector<real>{ 0.0, 0.0 },   // a
        std::vector<bool>{ true, true }, // ia
        real { 0.0 },                    // tol
        idx { 1000 }                     // MAXIT
        },
        FitmrqInvalidArgsCase{ "non_finite_sig",
        linear_model_jac,
        std::vector<real>{ 0, 1, 2 },    // x
        std::vector<real>{ 1, 3, 5 }, // y
        std::vector<real>{ 1, 1, std::numeric_limits<real>::quiet_NaN() },    // sig
        std::vector<real>{ 0.0, 0.0 },   // a
        std::vector<bool>{ true, true }, // ia
        real { 0.0 },                    // tol
        idx { 1000 }                     // MAXIT
        },
        FitmrqInvalidArgsCase{ "non_finite_a",
        linear_model_jac,
        std::vector<real>{ 0, 1, 2 },    // x
        std::vector<real>{ 1, 3, 5 }, // y
        std::vector<real>{ 1, 1, 1 },    // sig
        std::vector<real>{ 0.0, std::numeric_limits<real>::quiet_NaN() },   // a
        std::vector<bool>{ true, true }, // ia
        real { 0.0 },                    // tol
        idx { 1000 }                     // MAXIT
        },
        FitmrqInvalidArgsCase{ "negative_sig",
        linear_model_jac,
        std::vector<real>{ 0, 1, 2 },    // x
        std::vector<real>{ 1, 3, 5 }, // y
        std::vector<real>{ 1, 1, -1 },    // sig
        std::vector<real>{ 0.0, 0.0 },   // a
        std::vector<bool>{ true, true }, // ia
        real { 0.0 },                    // tol
        idx { 1000 }                     // MAXIT
        },
        FitmrqInvalidArgsCase{ "zero_sig",
        linear_model_jac,
        std::vector<real>{ 0, 1, 2 },    // x
        std::vector<real>{ 1, 3, 5 }, // y
        std::vector<real>{ 1, 1, 0 },    // sig
        std::vector<real>{ 0.0, 0.0 },   // a
        std::vector<bool>{ true, true }, // ia
        real { 0.0 },                    // tol
        idx { 1000 }                     // MAXIT
        }
    ),
    [](const ::testing::TestParamInfo<FitmrqInvalidArgsCase>& info) { return info.param.name; }
);

namespace {

inline constexpr real TOL_PARAM = 1e-9;   // ~2e-12
inline constexpr real TOL_COVAR = 1e6 * real_EPS;   // ~2e-10

void ExpectClose(const linalg::Matrix<real>& act, const linalg::Matrix<real>& exp, const real tol, std::string_view name)
{
    for (idx i {}; i < exp.length(); i++)
    {
        SCOPED_TRACE(std::format("{}: [{}, {}]", name, exp.ridx(i), exp.cidx(i)));
        EXPECT_NEAR(exp[i], act[i], std::max(tol, std::abs(exp[i])*tol));
    }
}

void ExpectClose(const std::span<const real> act, const std::span<const real> exp, const real tol, std::string_view name)
{
    for (idx i {}; i < exp.size(); i++)
    {
        SCOPED_TRACE(std::format("{}: [{}]", name, i));
        EXPECT_NEAR(exp[i], act[i], std::max(tol, std::abs(exp[i])*tol));
    }
}

void ExpectClose(const real act, const real exp, const real tol, std::string_view name)
{
    SCOPED_TRACE(name);
    EXPECT_NEAR(act, exp, std::max(tol, std::abs(exp)*tol));
}

void ExpectSymmetric(const linalg::Matrix<real>& A, std::string_view name)
{
    for (idx i {}; i < A.rows(); i++)
        for (idx j {}; j < A.cols(); j++)
        {
            SCOPED_TRACE(name);
            EXPECT_DOUBLE_EQ((A[i,j]), (A[j,i])) << std::format("Not Symmetric [{},{}] != [{},{}]", i, j, j, i);
        }
}

template <ArgsValJac F>
void ExpectGradientNull(F&& func, const std::vector<real>& x,
                        const std::vector<real>& y, const std::vector<real>& sig,
                        const std::vector<bool>& ia, const std::vector<real>& a_res)
{
    // Recompute Alpha, Beta based on results
    linalg::Matrix<real> alpha(a_res.size(), a_res.size());
    std::vector<real> beta(a_res.size(), 0.0);
    real chisq {};
    detail::mrqcof(func, x, y, sig, a_res, alpha, beta, ia, chisq);

    // Measure Gradient
    for (idx j = 0, l = 0; l < a_res.size(); ++l) {
        if (!ia[l]) continue;
        SCOPED_TRACE(std::format("free param l={} (slot {})", l, j));
        EXPECT_LE(std::abs(beta[j]) / std::sqrt(alpha[j, j]), 1e-4);
        ++j;
    }
}

}


//==============================================================================
// ORACLE TESTS
//==============================================================================
TEST(FitmrqTest, Linear_AllFree)
{
    // The Easiest Case for fitmrq

    // Inputs
    const std::vector<real> x {0.0, 1.0, 2.0, 3.0, 4.0};
    const std::vector<real> y {2.0, 3.0, 3.0, 7.0, 10.0};
    const std::vector<real> sig {1.0, 0.5, 1.0, 0.5, 1.0};
    const std::vector<bool> ia { true, true };
    const std::vector<real> a { 0.0, 0.0 };

    // Outputs (expected)
    const std::vector<real> a_expected { 1.0, 2.0 };
    const linalg::Matrix<real> covar_expected {
        { 15.0/44.0, -0.125 },
        {-0.125, 0.0625 }
    };
    const real chisq { 6.0 };

    FitResult res { fitmrq(linear_model_jac, x, y, sig, a, ia) };

    ExpectClose(res.covar, covar_expected, TOL_COVAR, "covar");
    ExpectClose(res.a, a_expected, TOL_PARAM, "param vec (a)");
    ExpectClose(res.chisq, chisq, TOL_PARAM, "chisq");
    ExpectSymmetric(res.covar, "covar symmetry");
    ExpectGradientNull(linear_model_jac, x, y, sig, ia, res.a);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.iterations, 20);
}

TEST(FitmrqTest, Quadratic_AllFree)
{
    // The Second Easiest Case for fitmrq

    // All Free
    // Inputs
    const std::vector<real> x   { -2.0, -1.0, 0.0, 1.0, 2.0 };
    const std::vector<real> y   { 10.0,  0.0, 1.0, 8.0, 16.0 };
    const std::vector<real> sig {  1.0,  1.0, 1.0, 1.0,  1.0 };
    const std::vector<bool> ia { true, true, true };
    const std::vector<real> a { 0.0, 0.0, 0.0 };

    // Outputs (expected)
    std::vector<real> a_expected { 1.0, 2.0, 3.0 };
    linalg::Matrix<real> covar_expected {
        { 17.0/35.0, 0.0, -1.0/7.0 },
        { 0.0, 1.0/10.0, 0 },
        { -1.0/7.0, 0.0, 1.0/14.0 },
    };
    const real chisq { 10 };

    FitResult res { fitmrq(quadratic_model_jac, x, y, sig, a, ia) };

    ExpectClose(res.covar, covar_expected, TOL_COVAR, "covar");
    ExpectClose(res.a, a_expected, TOL_PARAM, "param vec (a)");
    ExpectClose(res.chisq, chisq, TOL_PARAM, "chisq");
    ExpectSymmetric(res.covar, "covar symmetry");
    ExpectGradientNull(quadratic_model_jac, x, y, sig, ia, res.a);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.iterations, 20);
}

TEST(FitmrqTest, Quadratic_HoldOne)
{
    // The Second Easiest Case for fitmrq

    // Hold Middle
    // Inputs
    const std::vector<real> x   { -2.0, -1.0, 0.0, 1.0, 2.0 };
    const std::vector<real> y   { 10.0,  0.0, 1.0, 8.0, 16.0 };
    const std::vector<real> sig {  1.0,  1.0, 1.0, 1.0,  1.0 };
    const std::vector<bool> ia { true, false, true };
    const std::vector<real> a { 0.0, 0.0, 0.0 };

    // Outputs (expected)
    std::vector<real> a_expected { 1.0, 0.0, 3.0 };
    linalg::Matrix<real> covar_expected {
        { 17.0/35.0, 0.0, -1.0/7.0 },
        { 0.0, 0.0, 0.0 },
        { -1.0/7.0, 0.0, 1.0/14.0 },
    };
    const real chisq { 50 };

    FitResult res { fitmrq(quadratic_model_jac, x, y, sig, a, ia) };

    ExpectClose(res.covar, covar_expected, TOL_COVAR, "covar");
    ExpectClose(res.a, a_expected, TOL_PARAM, "param vec (a)");
    ExpectClose(res.chisq, chisq, TOL_PARAM, "chisq");
    ExpectSymmetric(res.covar, "covar symmetry");
    ExpectGradientNull(quadratic_model_jac, x, y, sig, ia, res.a);
    EXPECT_TRUE(res.converged);
    EXPECT_LE(res.iterations, 20);
}

TEST(FitmrqTest, SigmaScaling)
{
    const std::vector<real> x   { 0.0, 1.0, 2.0, 3.0, 4.0 };
    const std::vector<real> y   { 2.0, 3.0, 3.0, 7.0, 10.0 };
    const std::vector<real> sig1{ 1.0, 0.5, 1.0, 0.5, 1.0 };
    const std::vector<bool> ia  { true, true };
    const std::vector<real> a0  { 0.0, 0.0 };

    constexpr real c { 2.0 };
    std::vector<real> sig2(sig1.size());
    std::transform(sig1.begin(), sig1.end(), sig2.begin(),
                   [](real s){ return c * s; });

    const FitResult r1 { fitmrq(linear_model_jac, x, y, sig1, a0, ia) };
    const FitResult r2 { fitmrq(linear_model_jac, x, y, sig2, a0, ia) };

    for (idx j = 0; j < 2; ++j) {
        SCOPED_TRACE(std::format("a[{}]", j));
        EXPECT_NEAR(r2.a[j], r1.a[j], std::abs(r1.a[j]) * TOL_PARAM);
    }

    EXPECT_NEAR(r2.chisq, r1.chisq / (c*c), std::abs(r1.chisq/(c*c)) * TOL_PARAM);

    for (idx i = 0; i < 2; ++i)
        for (idx j = 0; j < 2; ++j) {
            SCOPED_TRACE(std::format("covar({},{})", i, j));
            EXPECT_NEAR((r2.covar[i,j]), (r1.covar[i,j] * c*c),
                        (std::abs(r1.covar[i,j] * c*c) * TOL_COVAR));
        }
}

//==============================================================================
// NUMERICAL JACOBIAN INTEGRATION
//==============================================================================
TEST(FitmrqTest, Numerical_Jacobian)
{
    // Use the chwirut2 nist function
    auto chwirut2 = [](real x, Params a)
    {
        const real E = std::exp(-a[0]*x);
        const real D = a[1] + a[2]*x;
        const real f = E / D;
        return f;
    };
    const real solver_tol { 1e-9 };
    const idx MAXIT { 1000 };

    // Unpack Nist Problem
    std::vector<real> x(nist::chwirut2.n, 0.0);
    std::transform(
        nist::chwirut2.xy.begin(), nist::chwirut2.xy.end(),
        x.begin(),
        [] (const auto& point) {
            return point.x;
        }
    );
    std::vector<real> y(nist::chwirut2.n, 0.0);
    std::transform(
        nist::chwirut2.xy.begin(), nist::chwirut2.xy.end(),
        y.begin(),
        [] (const auto& point) {
            return point.y;
        }
    );
    const idx ma { nist::chwirut2_data::start_a1.size() };
    const std::vector<real> sig(nist::chwirut2.n, 1.0);
    const std::vector<real> a(nist::chwirut2_data::start_a1.begin(), nist::chwirut2_data::start_a1.end());
    const std::vector<bool> ia(ma, true);

    // Solve with analtyical jacobian
    const FitResult r_an { fitmrq(nist::chwirut2.func, x, y, sig, a, ia, solver_tol, MAXIT) };

    // Solve with numerical jacobian
    const FitResult r_num { fitmrq(chwirut2, x, y, sig, a, ia, solver_tol, MAXIT) };

    // // Tests
    ExpectClose(r_num.a, r_an.a, 1e-8, "Numerical a vs. Analytical a");
    ExpectClose(r_num.covar, r_an.covar, 1e-10, "Numerical covar vs. Analytical covar");
    ExpectClose(r_num.chisq, r_an.chisq, 1e-12, "Numerical chisq vs Analytical chisq");
}

//==============================================================================
// NIST STrD Tests
//==============================================================================
struct FitmrqNISTCase {
    std::string name;              // for readable test IDs
    // Inputs
    nist::NistProblem problem;
    std::span<const real> start;
    real tol_solver;
    idx maxit;
    real tol_param;
    real tol_covar;
};
void PrintTo(const FitmrqNISTCase& c, std::ostream* os) { *os << c.name; }

class FitmrqNISTTest : public ::testing::TestWithParam<FitmrqNISTCase> {};

namespace {

void ExpectMatchesNist(std::string_view name,
                       const FitResult& res,
                       const nist::NistProblem& p,
                       real tol_param,
                       real tol_covar)
{
    SCOPED_TRACE(name);

    const real dof       { static_cast<real>(p.n - p.p) };
    const real chisq_exp { p.residual_sd * p.residual_sd * dof };

    EXPECT_NEAR(res.chisq, chisq_exp, std::abs(chisq_exp) * tol_param);

    for (idx j = 0; j < p.p; ++j)
    {
        SCOPED_TRACE(std::format("a[{}]", j));
        EXPECT_NEAR(res.a[j], p.certified_a[j],
                    std::abs(p.certified_a[j]) * tol_param);
    }

    for (idx j = 0; j < p.p; ++j)
    {
        SCOPED_TRACE(std::format("sd(a[{}])", j));
        const real sd { std::sqrt(res.covar[j, j] * res.chisq / dof) };
        EXPECT_NEAR(sd, p.certified_sd[j],
                    std::abs(p.certified_sd[j]) * tol_covar);
    }
}

}

TEST_P(FitmrqNISTTest, NIST_Test)
{
    // Get Test Parameters
    const FitmrqNISTCase& c = GetParam();
    const nist::NistProblem& p = c.problem;

    // Unpack Nist Problem
    std::vector<real> x(p.n, 0.0);
    std::transform(
        p.xy.begin(), p.xy.end(),
        x.begin(),
        [] (const auto& point) {
            return point.x;
        }
    );
    std::vector<real> y(p.n, 0.0);
    std::transform(
        p.xy.begin(), p.xy.end(),
        y.begin(),
        [] (const auto& point) {
            return point.y;
        }
    );
    const std::vector<real> sig(p.n, 1.0);
    const std::vector<real> a(c.start.begin(), c.start.end());
    const std::vector<bool> ia(p.p, true);
    
    // Call fitmrq
    const FitResult res { fitmrq(p.func, x, y, sig, a, ia, c.tol_solver, c.maxit) };

    // Tests
    ExpectMatchesNist(c.name, res, p, c.tol_param, c.tol_covar);
}

// Define Test Cases
INSTANTIATE_TEST_SUITE_P(
    NIST_STrD,
    FitmrqNISTTest,
    ::testing::Values(
        FitmrqNISTCase{ "nist_misra1a_start1",
            nist::misra1a,
            nist::misra1a_data::start_a1,
            real { 1e-12 },    // solver_tol
            idx  { 1000 },     // maxit
            real { 1e-10 },    // tol_param
            real { 1e-10 }     // tol_covar
        },
        FitmrqNISTCase{ "nist_misra1a_start2",
            nist::misra1a,
            nist::misra1a_data::start_a2,
            real { 1e-12 },   // solver_tol
            idx  { 1000 },    // maxit
            real { 1e-8 },    // tol_param (this is the lowest performing test)
            real { 1e-8 }     // tol_covar
        },
        FitmrqNISTCase{ "nist_chwirut2_start1",
            nist::chwirut2,
            nist::chwirut2_data::start_a1,
            real { 1e-12 },   // solver_tol
            idx  { 1000 },    // maxit
            real { 1e-9 },    // tol_param
            real { 1e-9 }     // tol_covar
        },
        FitmrqNISTCase{ "nist_chwirut2_start2",
            nist::chwirut2,
            nist::chwirut2_data::start_a2,
            real { 1e-12 },   // solver_tol
            idx  { 1000 },    // maxit
            real { 1e-10 },    // tol_param
            real { 1e-10 }     // tol_covar
        },
        FitmrqNISTCase{ "nist_gauss1_start1",
            nist::gauss1,
            nist::gauss1_data::start_a1,
            real { 1e-12 },   // solver_tol
            idx  { 1000 },    // maxit
            real { 1e-10 },    // tol_param
            real { 1e-10 }     // tol_covar
        },
        FitmrqNISTCase{ "nist_gauss1_start2",
            nist::gauss1,
            nist::gauss1_data::start_a2,
            real { 1e-12 },   // solver_tol
            idx  { 1000 },    // maxit
            real { 1e-10 },    // tol_param
            real { 1e-10 }     // tol_covar
        }
    ),
    [](const ::testing::TestParamInfo<FitmrqNISTCase>& info) { return info.param.name; }
);