#pragma once

#include <stator/core/types.h>

using namespace stator::core;

namespace stator::physics {
    // Good Initial guesses for ball timing model
    constexpr real a_i { 0.018 };
    constexpr real b_i { 1.8 };
    constexpr real co_i { -0.5 };
}