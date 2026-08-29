#pragma once
#include <compare>

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


}