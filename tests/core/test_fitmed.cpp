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

#include <stator/core/fitmed.h>
#include <stator/core/types.h>

using namespace stator::core;

//==============================================================================
// Rofunc Tests
//==============================================================================
namespace rofunc_test {
    const std::vector<real> x_base {1,2,3,4,5}; 
    const std::vector<real> y_base {3,5,7,9,11};
    std::vector<real> buf_base(x_base.size(), 0.0);

    const std::vector<real> x_even {1,2,3,4}; 
    const std::vector<real> y_even {2,4,6,8};
    std::vector<real> buf_even(x_even.size(), 0.0);
}

TEST(Rofunc, BaseB2)
{
    using namespace rofunc_test;

    const real b = 2.0;
    real abdev {};
    real a {};

    real sum { detail::rofunc(x_base, y_base, buf_base, a, b, abdev) };

    EXPECT_DOUBLE_EQ(0.0, sum);
    EXPECT_DOUBLE_EQ(0.0, abdev);
}

TEST(Rofunc, BaseB1)
{
    using namespace rofunc_test;

    const real b = 1.0;
    real abdev {};
    real a {};

    real sum { detail::rofunc(x_base, y_base, buf_base, a, b, abdev) };

    EXPECT_DOUBLE_EQ(6.0, sum);
    EXPECT_DOUBLE_EQ(6.0, abdev);
}

TEST(Rofunc, BaseB3)
{
    using namespace rofunc_test;

    const real b = 3.0;
    real abdev {};
    real a {};

    real sum { detail::rofunc(x_base, y_base, buf_base, a, b, abdev) };

    EXPECT_DOUBLE_EQ(-6.0, sum);
    EXPECT_DOUBLE_EQ(6.0, abdev);
}

TEST(Rofunc, EvenB1)
{
    using namespace rofunc_test;

    const real b = 1.0;
    real abdev {};
    real a {};

    real sum { detail::rofunc(x_even, y_even, buf_even, a, b, abdev) };

    EXPECT_DOUBLE_EQ(4.0, sum);
    EXPECT_DOUBLE_EQ(4.0, abdev);
}

TEST(Rofunc, OutlierB1)
{
    using namespace rofunc_test;

    const real b = 1.0;
    real abdev {};
    real a {};

    std::vector<real> y_outlier(y_base.begin(), y_base.end());
    y_outlier[4] = 100.0;

    real sum { detail::rofunc(x_base, y_outlier, buf_base, a, b, abdev) };

    EXPECT_DOUBLE_EQ(6.0, sum);
    EXPECT_DOUBLE_EQ(95.0, abdev);
}

TEST(Rofunc, N1)
{
    using namespace rofunc_test;

    const real b = 69.420;
    real abdev {};
    real a {};

    const std::vector<real> x { 2.0 };
    const std::vector<real> y { 7.0 };
    std::vector<real> buf(x.size(), 0.0);

    real sum { detail::rofunc(x, y, buf, a, b, abdev) };

    EXPECT_DOUBLE_EQ(0.0, sum);
    EXPECT_DOUBLE_EQ(0.0, abdev);
}

//==============================================================================
// INVARIANTS
//==============================================================================
namespace {
    real L1_S(std::vector<real>& x, std::vector<real>& y, const real a, const real b)
    {
        real sum {};
        for (idx j {}; j < x.size(); j++)
            sum += std::abs(y[j] - a - b*x[j]);
        return sum;
    }
}
TEST(FitmedTest, Invariants)
{
    // Test Data
    real h { 0.01 };
    std::vector<real> S1_x {1,2,3,4,5,6,7};
    std::vector<real> S1_y {2,4,5,8,9,11,12};
    std::vector<real> S2_x {1,2,3,4,5,6};
    std::vector<real> S2_y {3,5,8,9,11,40};

    // Result
    const LinRegResult res_S1 { fitmed(S1_x, S1_y) };
    const real S1_L1 { L1_S(S1_x, S1_y, res_S1.a, res_S1.b) };
    const LinRegResult res_S2 { fitmed(S2_x, S2_y) };
    const real S2_L1 { L1_S(S2_x, S2_y, res_S2.a, res_S2.b) };

    // Invariant 1: abdev
    SCOPED_TRACE("abdev invariant");
    EXPECT_DOUBLE_EQ(res_S1.abdev, S1_L1 / static_cast<real>(S1_x.size()));
    EXPECT_DOUBLE_EQ(res_S2.abdev, S2_L1 / static_cast<real>(S2_x.size()));

    // Invariant 2: Preturbed Parameters are Worse L1 Norm than ours
    SCOPED_TRACE("preturbed parameter invariant");
    // S1
    EXPECT_LE(S1_L1, L1_S(S1_x, S1_y, res_S1.a+h, res_S1.b) + 1e-9);
    EXPECT_LE(S1_L1, L1_S(S1_x, S1_y, res_S1.a-h, res_S1.b) + 1e-9);
    EXPECT_LE(S1_L1, L1_S(S1_x, S1_y, res_S1.a, res_S1.b+h) + 1e-9);
    EXPECT_LE(S1_L1, L1_S(S1_x, S1_y, res_S1.a, res_S1.b-h) + 1e-9);
    // S2
    EXPECT_LE(S2_L1, L1_S(S2_x, S2_y, res_S2.a+h, res_S2.b) + 1e-9);
    EXPECT_LE(S2_L1, L1_S(S2_x, S2_y, res_S2.a-h, res_S2.b) + 1e-9);
    EXPECT_LE(S2_L1, L1_S(S2_x, S2_y, res_S2.a, res_S2.b+h) + 1e-9);
    EXPECT_LE(S2_L1, L1_S(S2_x, S2_y, res_S2.a, res_S2.b-h) + 1e-9);

    // Invariant 3: x_size == odd should have one residual == 0 
    SCOPED_TRACE("odd size zero residual invariant");
    std::vector<real> s1_residual(S1_x.size(), 0.0);
    std::transform(S1_x.begin(),
                   S1_x.end(), 
                   S1_y.begin(),
                   s1_residual.begin(),
                   [&res_S1](real x, real y) { return std::abs(y - res_S1.a - res_S1.b*x); });
    EXPECT_NEAR(*(std::min_element(s1_residual.begin(), s1_residual.end())), 0.0, 1e-12);

    // Invariante 4: Translational Invariance
    SCOPED_TRACE("translational invariance");
    const real tx = 10.0;
    const real ty = 100.0;
    std::vector<real> S1_y_trans(S1_y.size(), 0.0);
    std::vector<real> S1_x_trans(S1_x.size(), 0.0);
    for (idx j {}; j < S1_x.size(); j++)
    {
        S1_y_trans[j] = S1_y[j] + ty;
        S1_x_trans[j] = S1_x[j] + tx;
    }
    LinRegResult res_tx { fitmed(S1_x_trans, S1_y) };
    LinRegResult res_ty { fitmed(S1_x, S1_y_trans) };

    EXPECT_NEAR(res_S1.abdev, res_tx.abdev, 1e-12) << "S1.abdev = tx.abdev";
    EXPECT_NEAR(res_S1.abdev, res_ty.abdev, 1e-12) << "S1.abdev = ty.abdev";
    EXPECT_NEAR(res_ty.b, res_S1.b, 1e-9) << "S1.b = ty.b";
    EXPECT_NEAR(res_tx.b, res_S1.b, 1e-9) << "S1.b = tx.b";
    EXPECT_NEAR(res_tx.a, res_S1.a - tx * res_S1.b, 1e-7) << "S1.a - 10*S1.b = tx.a";
    EXPECT_NEAR(res_ty.a, res_S1.a + ty, 1e-7) << "S1.a + 100 = ty.a";
}

//==============================================================================
// INVALID ARGUMENT
//==============================================================================
TEST(FitmedTest, Invalid_Argument)
{
    const std::vector<real> x { 1.0, 2.0 };
    const std::vector<real> y { 1.0, 2.0, 3.0 };
    EXPECT_THROW(fitmed(x, y), InvalidArgument);
}

//==============================================================================
// Fitmed Expected Values
//==============================================================================
struct FitmedExpectedValueCase {
    std::string name;              // for readable test IDs
    // Inputs
    std::vector<real> x;
    std::vector<real> y;
    real tol_a;
    real tol_b;
    real tol_abdev;
    // Outputs
    real a_expected;
    real b_expected;
    real abdev_expected;
};
void PrintTo(const FitmedExpectedValueCase& c, std::ostream* os) { *os << c.name; }

class FitmedExpectedValueTest : public ::testing::TestWithParam<FitmedExpectedValueCase> {};


TEST_P(FitmedExpectedValueTest, ExpectedValue)
{
    // Get Test Parameters
    const FitmedExpectedValueCase& c = GetParam();
    
    // Call fitmrq
    const LinRegResult res { fitmed(c.x, c.y) };

    // Tests
    EXPECT_NEAR(c.a_expected, res.a, c.tol_a);
    EXPECT_NEAR(c.b_expected, res.b, c.tol_b);
    EXPECT_NEAR(c.abdev_expected, res.abdev, c.tol_abdev);
}

// Define Test Cases
INSTANTIATE_TEST_SUITE_P(
    Expected_Values,
    FitmedExpectedValueTest,
    ::testing::Values(
        FitmedExpectedValueCase{ "test_1",
            std::vector<real> {1,2,3,4,5},      // x
            std::vector<real> {3,5,7,9,11},     // y
            real {1e-9},                        // tol_a
            real {1e-7},                        // tol_b
            real {1e-12},                       // tol_abdev
            real {1.0},                         // a_expected
            real {2.0},                         // b_expected
            real {0.0}                          // abdev_expected
        },
        FitmedExpectedValueCase{ "test_2",
            std::vector<real> {1,2,3,4},      // x
            std::vector<real> {2,4,6,8},     // y
            real {1e-9},                        // tol_a
            real {1e-7},                        // tol_b
            real {1e-12},                       // tol_abdev
            real {0.0},                         // a_expected
            real {2.0},                         // b_expected
            real {0.0}                          // abdev_expected
        },
        FitmedExpectedValueCase{ "test_3",
            std::vector<real> {1,2,3,4,5},      // x
            std::vector<real> {11,9,7,5,3},     // y
            real {1e-9},                        // tol_a
            real {1e-7},                        // tol_b
            real {1e-12},                       // tol_abdev
            real {13.0},                         // a_expected
            real {-2.0},                         // b_expected
            real {0.0}                          // abdev_expected
        },
        FitmedExpectedValueCase{ "test_4",
            std::vector<real> {1,2,3,4,5},      // x
            std::vector<real> {5,5,5,5,5},      // y
            real {1e-9},                        // tol_a
            real {1e-7},                        // tol_b
            real {1e-12},                       // tol_abdev
            real {5.0},                         // a_expected
            real {0.0},                         // b_expected
            real {0.0}                          // abdev_expected
        },
        FitmedExpectedValueCase{ "test_5",
            std::vector<real> {1,2,3,4,5},      // x
            std::vector<real> {3,5,7,9,100},    // y
            real {1e-9},                        // tol_a
            real {1e-7},                        // tol_b
            real {1e-9},                        // tol_abdev
            real {1.0},                         // a_expected
            real {2.0},                         // b_expected
            real {89.0/5.0}                     // abdev_expected
        },
        FitmedExpectedValueCase{ "test_6",
            std::vector<real> {1,2,3,4,5},      // x
            std::vector<real> {3,5,7,9,-50.0},  // y
            real {1e-9},                        // tol_a
            real {1e-7},                        // tol_b
            real {1e-9},                        // tol_abdev
            real {1.0},                         // a_expected
            real {2.0},                         // b_expected
            real {61.0/5.0}                     // abdev_expected
        }
    ),
    [](const ::testing::TestParamInfo<FitmedExpectedValueCase>& info) { return info.param.name; }
);
