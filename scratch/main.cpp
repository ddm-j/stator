#include <utility>
#include <format>

#include <cmath>

#include <stator/core/fitmrq.h>
#include <stator/core/numeric_result.h>
#include <stator/core/types.h>
#include <stator/core/concepts.h>
#include <stator/core/constants.h>

int main()
{
    using namespace stator::core;

    auto ball_timing_model  = [] (real x, Params params)
    {
        const real a { params[0] };
        const real b { params[1] };
        const real co { params[2] };
        const real f { 
            (1.0 / (a * b)) * (co - std::asinh(std::sinh(co) * std::exp(2*a*x*pi)))
        };
        return f;
    };
    
    const std::vector<real> k { 1, 2, 3, 4, 5 };
    const std::vector<real> laps { 1.5, 1.76, 1.83, 2.06, 2.2 };
    std::vector<real> tk(laps.size(), 0.0);
    std::partial_sum(laps.begin(), laps.end(), tk.begin());
    std::cout << std::format("tk.size() = {}\n\n\n", tk.size());
    // const std::vector<real> sig(k.size(), 1.0);

    const std::vector<real> params { 0.018, 1.8, -0.5 };
    // const std::vector<bool> ia(params.size(), true);
    FitResult res { fitmrq(ball_timing_model, k, tk, 1.0, params, 1e-8) };

    for (idx j {}; j < params.size(); j++)
    {
        std::cout << std::format("param[{}] = {}\n", j, res.a[j]);
    }


    return 1;
}