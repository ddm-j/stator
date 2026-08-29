// concepts.h
#pragma once

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

}
