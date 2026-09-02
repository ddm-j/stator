// types.h
#pragma once

#include <cstddef>
#include <limits>
#include <vector>

namespace stator::core {
    // Vocabulary and Type Aliases

    // Numeric
    using real = double; // Cannot be less precise than "double"
    using idx = std::size_t;
    using size = std::size_t;

    // Storage
    using ParamVec = std::vector<real>;


}