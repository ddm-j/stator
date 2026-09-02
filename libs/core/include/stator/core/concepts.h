// concepts.h
#pragma once

#include <vector>
#include <span>
#include <utility>
#include "stator/core/types.h"

namespace stator::core{

// Concept Arguments
using Params     = const std::vector<real>&;
using Derivative = std::vector<real>&;


// f(x; a) — Model f(x, a, b, c...)
template <class F>
concept Model =
    std::regular_invocable<F, real, Params> &&
    std::same_as<std::invoke_result_t<F, real, Params>, real>;

// f(x, a) - f(x, a) where [f(x), dfdx(x)] is returned as a pair
template <class F>
concept DifferentiableModel = 
    std::regular_invocable<F, real, Params> &&
    std::same_as<std::invoke_result_t<F, real, Params>, std::pair<real, real>>;

// f(x, a, dadx) - f(x, a) where dadx is a vector modified by the callable and stores da/dx
template <class F>
concept SensitivityModel =
    std::regular_invocable<F, real, Params, Derivative> &&
    std::same_as<std::invoke_result_t<F, real, Params, Derivative>, real>;

}
