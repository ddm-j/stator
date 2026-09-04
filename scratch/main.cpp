#include <utility>
#include <format>

#include <cmath>

#include <stator/physics/rim.h>
#include <stator/core/fitmrq.h>
#include <stator/core/golden_section.h>
#include <stator/core/numeric_result.h>
#include <stator/core/types.h>
#include <stator/core/concepts.h>
#include <stator/core/constants.h>

int main()
{
    using namespace stator::core;
    using namespace stator::physics;
    
    const std::vector<real> k { 0, 1, 2, 3, 4, 5 };
    const std::vector<real> laps { 0.0, 1.5, 1.76, 1.83, 2.06, 2.2 };
    std::vector<real> tk(laps.size(), 0.0);
    std::partial_sum(laps.begin(), laps.end(), tk.begin());

    const std::vector<real> params { 0.018, 1.8, -0.5 };
    // const std::vector<bool> ia(params.size(), true);

    // Initial Parameter Fit
    const real human_error {0.02 }; // 20ms human timing error
    FitResult res { fitmrq(ball_timing_model, k, tk, human_error, params, 1e-8) };

    std::cout << "LM Fit:\n";
    for (idx j {}; j < params.size(); j++)
    {
        std::cout << std::format("param[{}] = {}\n", j, res.a[j]);
    }

    // Parameter Refinement
    auto refinement_model = make_ball_timing_refinement_model(res.a[1], tk);
    // Setup an initial bracket
    real sig_a { std::sqrt(res.covar[0, 0]) };
    real aLeft { std::max(1e-9, res.a[0] - sig_a)};
    real aRight { res.a[0] + sig_a };
    std::cout << std::format("a bracket: [{}, {}]\n", aLeft, aRight);

    // // Test brackets
    // std::cout << "\n\nS(a) bracket sweep debug\n";
    // for (idx i {}; i < 10; i++)
    // {
    //     real a { aLeft + static_cast<real>(i)*(aRight - aLeft) };
    //     real Sa { refinement_model(a, dummy) };
    //     std::cout << std::format("S({}) = {}\n", a, Sa);
    // }

    MinResult1D min { golden_section(refinement_model, aLeft, aRight) };
    std::cout << std::format("gold_sec iterations: {} \ngold_sec converged: {}\n", min.iterations, min.converged);

    // Final Parameters
    std::cout << "Parameters after refinement: \n";
    std::cout << std::format("a = {} \nb = {}\n", min.x, res.a[1]);

    return 1;
}