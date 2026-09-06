#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

#include <stator/core/types.h>

namespace stator::core {

inline std::vector<real> logspace(real start, real end, idx n, real base = 10.0) {
    std::vector<real> result;
    if (n <= 0) return result;
    result.resize(n);
    if (n == 1) {
        result[0] = std::pow(base, end);
        return result;
    }
    double step = (end - start) / (static_cast<real>(n) - 1);
    for (idx i = 0; i < n; ++i) {
        result[i] = std::pow(base, start + static_cast<real>(i) * step);
    }
    return result;
}

template <typename T>
idx argmin(const std::vector<T>& arg)
{
    if (arg.empty()) return 0;
    auto min_it { std::min_element(arg.begin(), arg.end()) };
    return static_cast<idx>(std::distance(arg.begin(), min_it));
}

}
