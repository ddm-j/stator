// constants.h
#pragma once

namespace stator::core {

    // Numeric Constants
    inline constexpr real real_EPS { std::numeric_limits<real>::epsilon() };

    // Bracketing Constants
    constexpr real GOLD { 1.618034 };
    constexpr real GLIMIT { 100.0 }; 
    constexpr real TINY { 1e-20 };

    // Golden-Section Search Constants
    inline constexpr real GOLD_R { 0.61803399 };
    inline constexpr real GOLD_C { 1.0 - GOLD_R };

}