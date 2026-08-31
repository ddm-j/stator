// concepts.h
#pragma once

#include <vector>
#include <span>
#include "stator/core/types.h"

namespace stator::core{

// Value-Only Callback (f(x) alone)
template <class F>
concept ValOnly = std::regular_invocable<F, real> &&
    std::convertible_to<std::invoke_result_t<F, real>, real>;

// Value-Derivative Pair Callback (implies f(x) and df(x) at the same point)
template <class F>
concept ValDer = std::regular_invocable<F, real> &&
    std::convertible_to<std::invoke_result_t<F, real>, std::pair<real, real>>;


using Params = const std::vector<real>&;
using Jac    = std::vector<real>&;

// f(x; a) only — Model f(x, a, b, c...)
template <class F>
concept ArgsVal =
    std::regular_invocable<F, real, Params> &&
    std::same_as<std::invoke_result_t<F, real, Params>, real>;

// f(x; a) and the jacobian
template <class F>
concept ArgsValJac =
    std::regular_invocable<F, real, Params, Jac> &&
    std::same_as<std::invoke_result_t<F, real, Params, Jac>, real>;

}
