#include <cstddef>
#include <climits>
#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <limits>
#include <format>

#include <linalg/Matrix.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <stator/core/fitmrq.h>

using namespace stator::core;
using linalg::Matrix;

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

// Symmetric Matrix Check
void test_symmetry(const linalg::Matrix<real>& A)
{
    for (idx i {}; i < A.rows(); i++)
        for (idx j {}; j < A.cols(); j++)
        {
            EXPECT_DOUBLE_EQ((A[i,j]), (A[j,i])) << std::format("Not Symmetric [{},{}] != [{},{}]", i, j, j, i);
        }
}

// Test Outpus
void test_results(
    const linalg::Matrix<real>& alpha,
    const linalg::Matrix<real>& alpha_expected,
    const std::vector<real>& beta,
    const std::vector<real>& beta_expected,
    const real chisq,
    const real chisq_expected
)
{
    // Alpha
    for (idx i {}; i < alpha.length(); i++)
    {
        EXPECT_DOUBLE_EQ(alpha[i], alpha_expected[i]) << std::format("alpha @ [{}, {}]", alpha.ridx(i), alpha.cidx(i));
    }
    // Beta
    for (idx i {}; i < beta.size(); i++)
    {
        EXPECT_DOUBLE_EQ(beta[i], beta_expected[i]) << std::format("beta @ [{}]", i);
    }
    // Chisq
    EXPECT_DOUBLE_EQ(chisq, chisq_expected) << "chisq not equal";
    // Symmetry
    test_symmetry(alpha);
}

//==============================================================================
// CASE SETUP
//==============================================================================
struct MrqcofCase {
    std::string name;              // for readable test IDs
    // Inputs
    real (*func) (real, const std::vector<real>&, std::vector<real>&);
    std::vector<real> x, y, sig;
    std::vector<real> a;
    std::vector<bool> ia;
    // Outputs
    linalg::Matrix<real> alpha_expected;
    std::vector<real> beta_expected;
    real chisq_expected;
};
void PrintTo(const MrqcofCase& c, std::ostream* os) { *os << c.name; }

class MrqcofTest : public ::testing::TestWithParam<MrqcofCase> {};

TEST_P(MrqcofTest, ReturnsExpected)
{
    // Get Test Parameters
    const MrqcofCase& c = GetParam();

    // Construct Alpha/Beta/Chisq
    linalg::Matrix<real> alpha(c.alpha_expected.rows(), c.alpha_expected.cols());
    std::vector<real> beta(c.beta_expected.size(), 0.0);
    real chisq {};

    // Call mrqcof
    detail::mrqcof(c.func, c.x, c.y, c.sig, c.a, alpha, beta, c.ia, chisq);

    // Run Tests
    test_results(alpha, c.alpha_expected, beta, c.beta_expected, chisq, c.chisq_expected);
}

INSTANTIATE_TEST_SUITE_P(
    Functions,
    MrqcofTest,
    ::testing::Values(
        MrqcofCase{ "linear_model_0params_unit_sigma",
        linear_model_jac,
        std::vector<real>{ 0, 1, 2 },    // x
        std::vector<real>{ 1, 3, 5 },    // y
        std::vector<real>{ 1, 1, 1 },    // sig
        std::vector<real>{ 0.0, 0.0 },   // a
        std::vector<bool>{ true, true }, // ia
        linalg::Matrix<real>{
                        {3, 3},          // alpha_expected
                        {3, 5}
                        },
        std::vector<real>{ 9, 13 },      // beta_expected
        real { 35 }                      // chisq_expected
        },
        MrqcofCase{ "linear_model_0params_nonunit_sigma",
        linear_model_jac,
        std::vector<real>{ 0, 1, 2 },    // x
        std::vector<real>{ 1, 3, 5 },    // y
        std::vector<real>{ 1, 2, 1 },    // sig
        std::vector<real>{ 0.0, 0.0 },   // a
        std::vector<bool>{ true, true }, // ia
        linalg::Matrix<real>{
                        {2.25, 2.25},    // alpha_expected
                        {2.25, 4.25}
                        },
        std::vector<real>{ 6.75, 10.75 },// beta_expected
        real { 28.25 }                   // chisq_expected
        },
        MrqcofCase{ "linear_model_0params_ia_right",
        linear_model_jac,
        std::vector<real>{ 0, 1, 2 },    // x
        std::vector<real>{ 1, 3, 5 },    // y
        std::vector<real>{ 1, 1, 1 },    // sig
        std::vector<real>{ 0.0, 0.0 },   // a
        std::vector<bool>{ true, false }, // ia
        linalg::Matrix<real>{ 3 },       // alpha_expected
        std::vector<real>{ 9 },          // beta_expected
        real { 35 }                      // chisq_expected
        },
        MrqcofCase{ "linear_model_0params_ia_left",
        linear_model_jac,
        std::vector<real>{ 0, 1, 2 },    // x
        std::vector<real>{ 1, 3, 5 },    // y
        std::vector<real>{ 1, 1, 1 },    // sig
        std::vector<real>{ 0.0, 0.0 },   // a
        std::vector<bool>{ false, true }, // ia
        linalg::Matrix<real>{ 5 },       // alpha_expected
        std::vector<real>{ 13 },          // beta_expected
        real { 35 }                      // chisq_expected
        },
        MrqcofCase{ "linear_model_perfect_fit",
        linear_model_jac,
        std::vector<real>{ 0, 1, 2 },    // x
        std::vector<real>{ 1, 3, 5 },    // y
        std::vector<real>{ 1, 1, 1 },    // sig
        std::vector<real>{ 1.0, 2.0 },   // a
        std::vector<bool>{ true, true }, // ia
        linalg::Matrix<real>{
                        {3, 3},          // alpha_expected
                        {3, 5}
                        },
        std::vector<real>{ 0, 0 },      // beta_expected
        real { 0 }                      // chisq_expected
        },
        MrqcofCase{ "quadratic_model_mirror",
        quadratic_model_jac,
        std::vector<real>{ 0, 1, 2 },         // x
        std::vector<real>{ 1, 3, 5 },         // y
        std::vector<real>{ 1, 1, 1 },         // sig
        std::vector<real>{ 0.0, 0.0, 0.0 },   // a
        std::vector<bool>{ true, true, true}, // ia
        linalg::Matrix<real>{
                        {3, 3, 5},            // alpha_expected
                        {3, 5, 9},
                        {5, 9, 17},
                        },
        std::vector<real>{ 9, 13, 23 },       // beta_expected
        real { 35 }                           // chisq_expected
        },
        MrqcofCase{ "quadratic_model_hole",
        quadratic_model_jac,
        std::vector<real>{ 0, 1, 2 },         // x
        std::vector<real>{ 1, 3, 5 },         // y
        std::vector<real>{ 1, 1, 1 },         // sig
        std::vector<real>{ 0.0, 0.0, 0.0 },   // a
        std::vector<bool>{ true, false, true},// ia
        linalg::Matrix<real>{
                        {3, 5},               // alpha_expected
                        {5, 17},
                        },
        std::vector<real>{ 9, 23 },       // beta_expected
        real { 35 }                           // chisq_expected
        },
        MrqcofCase{ "nonpolynomial_model",
        non_polynomial_jac,
        std::vector<real>{ 1, 2, 4 },         // x
        std::vector<real>{ 1, 3, 5 },         // y
        std::vector<real>{ 1, 1, 1 },         // sig
        std::vector<real>{ 0.0, 0.0, 0.0 },   // a
        std::vector<bool>{ true, true, true},// ia
        linalg::Matrix<real>{
                        {3, 7, 1.3125},               // alpha_expected
                        {7, 21, 1.75},
                        {1.3125, 1.75, 1.06640625}
                        },
        std::vector<real>{ 9, 27, 2.0625 },       // beta_expected
        real { 35 }                           // chisq_expected
        },
        MrqcofCase{ "nonpolynomial_model_hole",
        non_polynomial_jac,
        std::vector<real>{ 1, 2, 4 },         // x
        std::vector<real>{ 1, 3, 5 },         // y
        std::vector<real>{ 1, 1, 1 },         // sig
        std::vector<real>{ 0.0, 0.0, 0.0 },   // a
        std::vector<bool>{ true, false, true},// ia
        linalg::Matrix<real>{
                        {3, 1.3125},               // alpha_expected
                        {1.3125, 1.06640625}
                        },
        std::vector<real>{ 9, 2.0625 },       // beta_expected
        real { 35 }                           // chisq_expected
        }
    ),
    [](const ::testing::TestParamInfo<MrqcofCase>& info) { return info.param.name; }
);