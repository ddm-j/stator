#include <utility>
#include <format>

#include <stator/core/rtsafe.h>

int main()
{
    using real = stator::core::real;
    using RootResult = stator::core::RootResult;

    RootResult root {};

    // Test Functions
    auto xsquared = [](real X) -> std::pair<real, real> {
        return { 
            X*X - 2, // f(x)
            2*X      // df(x)
         };
    };
    auto xcubed = [](real X) -> std::pair<real, real> {
        return { 
            X*X*X - X - 2, // f(x)
            3*X*X - 1      // df(x)
         };
    };
    auto hostile_xcubed = [](real X) -> std::pair<real, real> {
        return { 
            X*X*X - 2*X + 2, // f(x)
            3*X*X - 2      // df(x)
         };
    };

    root = stator::core::rtsafe(xsquared, 0.0, 2.0);
    std::cout << std::format("The root of X^2 - 2 = {}\n", root.x) << std::endl;

    root = stator::core::rtsafe(xcubed, 1.0, 2.0);
    std::cout << std::format("The root of X^3 - X - 2 = {}\n", root.x) << std::endl;

    root = stator::core::rtsafe(hostile_xcubed, -2.0, 2.0);
    std::cout << std::format("The root of X^3 - 2X + 2 = {}\n", root.x) << std::endl;
}