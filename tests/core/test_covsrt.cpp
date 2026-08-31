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

//==============================================================================
// Covsrt Tests
//==============================================================================
TEST(Covrst, Identity)
{
    std::vector<bool> ia { true, true, true };
    Matrix<real> covar {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    auto covar_orig = covar;

    detail::covsrt(covar, ia);

    EXPECT_EQ(covar, covar_orig);
}

TEST(Covrst, Zeroing)
{
    std::vector<bool> ia { true, true, false }; // No permutations, expect 2x2 of data, rest zeros
    Matrix<real> covar {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    Matrix<real> expected {
        {1, 2, 0},
        {4, 5, 0},
        {0, 0, 0}
    };

    detail::covsrt(covar, ia);

    EXPECT_EQ(covar, expected);
}

TEST(Covrst, Swapping)
{
    std::vector<bool> ia { true, false, true }; // Forces a permute
    Matrix<real> covar {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    Matrix<real> expected {
        {1, 0, 2},
        {0, 0, 0},
        {4, 0, 5}
    };

    detail::covsrt(covar, ia);

    EXPECT_EQ(covar, expected);
}
