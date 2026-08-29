#include <utility>
#include <format>

#include <stator/core/golden_section.h>

int main()
{
    using real = stator::core::real;
    using MinResult1D = stator::core::MinResult1D;

    MinResult1D res {};

    // Test Functions
    auto xsquared = [](real X) -> real {
        return X*X - 2;
    };

    res = stator::core::golden_section(xsquared, 0.0, 2.0);
    std::cout << std::format("The minimum of X^2 - 2 = {}\n", res.x) << std::endl;

}