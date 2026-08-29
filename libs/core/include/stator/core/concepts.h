// concepts.h
#pragma once

#include "stator/core/types.h"

namespace stator::core{

// Value-Derivative Pair (Callable)
template <class F>
concept ValDer = std::regular_invocable<F, stator::core::real> &&
    std::convertible_to<std::invoke_result_t<F, stator::core::real>, std::pair<stator::core::real, stator::core::real>>;

}
