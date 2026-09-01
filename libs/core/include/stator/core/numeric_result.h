#pragma once

#include <compare>

#include <linalg/Matrix.h>

namespace stator::core {

// Generic Numeric Result
struct NumericResult
{
    idx iterations {};
    bool converged {false};
};

// RootFinding Result
struct RootResult : NumericResult
{
    real x {};
    real f {};
    real df {};
};

// MinResult1D
struct MinResult1D : NumericResult
{
    real x {};
    real f {};
    real width {};
};

// MinResult1D
struct FitResult : NumericResult
{
    std::vector<real> a {};         // Parameter Vector
    linalg::Matrix<real> covar;     // Parameter Covariance
    std::vector<real> beta {};      // Residuals
    real chisq {};                  // Chisq
};

}