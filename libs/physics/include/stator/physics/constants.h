#pragma once

#include <stator/core/types.h>
#include <array>
#include <string>

using namespace stator::core;

namespace stator::physics {
    // Good Initial guesses for ball timing model
    constexpr real a_i { 0.018 };
    constexpr real b_i { 1.8 };
    constexpr real c0_i { -0.5 };

    // Human Timing Error
    constexpr real sig_human { 0.035 }; // in seconds, emperically measured

    // Roulette Wheel Layouts
    constexpr std::array<std::string_view, 37> european_wheel{
        "0", "32", "15", "19", "4", "21", "2", "25", "17", "34", 
        "6", "27", "13", "36", "11", "30", "8", "23", "10", "5",
        "24", "16", "33", "1", "20", "14", "31", "9", "22", "18",
        "29", "7", "28", "12", "35", "3", "26"
    };
    constexpr std::array<std::string_view, 38> american_wheel{
        "0", "28", "9", "26", "30", "11", "7", "20", "32", "17",
        "5", "22", "34", "15", "3", "24", "36", "13", "1", "00",
        "27", "10", "25", "29", "12", "8", "19", "31", "18", "6",
        "21", "33", "16", "4", "23", "35", "14", "2"
    };
}