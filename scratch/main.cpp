#include <utility>
#include <format>

#include <cmath>

#include <stator/physics/predict.h>
#include <stator/physics/ball.h>
#include <stator/physics/departure.h>
#include <stator/physics/constants.h>
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

    FitParams params {
        BallParams(0.0225, 1.8257419),
        DepartureParams(0.0, 0.2, 7.621509)
    };
    const real To1 { 1.480553 };
    const real To2 { 0.895641 };

    auto theta1 { predict_theta(To1, params) };
    if (theta1.has_value())
        std::cout << std::format("Case 1: theta = {}\n", theta1.value());
    else
        std::cout << "Failed to find departure theta\n";

    auto theta2 { predict_theta(To2, params) };
    if (theta2.has_value())
        std::cout << std::format("Case 1: theta = {}\n", theta2.value());
    else
        std::cout << "Failed to find departure theta\n";

    // // Sample Data (raw, needs first time subtracted off)
    // bool DEGENERATE { true };
    // std::vector<std::vector<real>> tk;
    // idx ndat;
    // std::vector<real> bin;
    // if (DEGENERATE)
    // {
    //     tk = {
    //         { 616.036, 617.716, 619.515, 621.302, 623.422, 625.646 },      // B0004  a=0.0101 b=6.64e-07  chisq=12.08
    //         { 979.578, 981.123, 982.808, 984.763, 986.829, 989.185 },      // B0008  a=0.0165 b=7.31e-05  chisq=1.48
    //         { 1102.218, 1103.811, 1105.494, 1107.333, 1109.376, 1111.693 },// B0010  a=0.0128 b=-0.0091   chisq=12.52
    //         { 1296.203, 1297.795, 1299.505, 1301.295, 1303.365, 1305.643 },// B0012  a=0.0127 b=0.00155   chisq=10.53
    //         { 1414.349, 1415.943, 1417.478, 1419.381, 1421.353, 1423.538 },// B0015  a=0.0104 b=0.00775   chisq=34.88
    //     };
    //     ndat = tk.size();
    //     bin = { 31, 29, 33, 34, 36 };
    // } else
    // {
    //      tk = {
    //         { 344.456, 345.949, 347.672, 349.554, 351.624, 353.973 },      // B0001  a=0.0254 b=1.834 chisq=1.60
    //         { 507.455, 508.892, 510.656, 512.543, 514.665, 516.912 },      // B0003  a=0.0512 b=2.650 chisq=2.29
    //         { 1020.058, 1021.546, 1023.229, 1025.060, 1027.083, 1029.307 },// B0009  a=0.0233 b=1.916 chisq=0.34
    //         { 1148.592, 1150.045, 1151.687, 1153.578, 1155.523, 1157.749 },// B0011  a=0.0302 b=2.319 chisq=2.65
    //         { 1335.245, 1336.767, 1338.449, 1340.306, 1342.336, 1344.571 },// B0013  a=0.0177 b=1.263 chisq=0.02
    //     };
    //     ndat = tk.size();
    //     bin = { 30, 1, 34, 35, 34 };
    // }
    // std::vector<real> theta_f(ndat, 0.0);

    // for (idx i {}; i < ndat; i++)
    // {
    //     const real ts0 { tk[i][0] };
    //     theta_f[i] = 2*pi*static_cast<real>(tk[i].size()-1) + 2*pi*(bin[i] - 0.5)/36.0;
    //     for(idx j{}; j < tk[i].size(); j++)
    //     {
    //         tk[i][j] -= ts0;
    //     }
    // }

    // // Fits
    // std::vector<real> as(ndat, 0.0);
    // std::vector<real> bs(ndat, 0.0);
    // const real sig { 0.035 }; // ball timing error (s)
    // std::cout << std::format("Fitting a,b parameters: \n");
    // for (idx j{}; j < ndat; j++)
    // {
    //     // Fit
    //     FitResult res { fit_ball_timings(tk[j], sig)};
    //     const real siga { std::sqrt(res.covar[0, 0]) };
        
    //     // Refine
    //     auto refinement_model = make_ball_timing_refinement_model(res.a[1], tk[j]);
    //     MinResult1D min { golden_section(refinement_model, res.a[0]-2*siga, res.a[0]+2*siga) };

    //     as[j] = min.x;
    //     bs[j] = res.a[1];
    //     std::cout << std::format("  it{} - a from {} -> to {}\n", j, res.a[0], min.x);
    // }

    // // Mean Parameters
    // real a { std::reduce(as.begin(), as.end()) / static_cast<real>(ndat) };
    // real b { std::reduce(bs.begin(), bs.end()) / static_cast<real>(ndat) };
    // std::cout << std::format("Mean Parameters: \na = {} \nb = {}\n\n", a, b);

    // // Fit Departure
    // std::vector<real> To(ndat, 0.0);
    // for (idx j {}; j < ndat; j++)
    //     To[j] = tk[j][1];

    // auto departure_model = make_departure_objective(a, b, To, theta_f);

    // /// // Coarse Scan to find global minimum
    // const idx N {15};
    // const real dphi { 2*pi/static_cast<real>(N) };
    // std::vector<real> S_coarse(N, 0.0);
    // for (idx i {}; i < N; i++)
    // {
    //     real phi { static_cast<real>(i) * dphi };
    //     S_coarse[i] = departure_model(phi).abdev;
    // }
    // auto min_it = std::min_element(S_coarse.begin(), S_coarse.end());
    // idx idx_min { static_cast<idx>(std::distance(S_coarse.begin(), min_it)) };
    // std::cout << "Idx of the coarse min: " << idx_min << "\n";

    // // // Hone the First Minimum
    // const MinResult1D min1_res { golden_section([&](real phi) { return departure_model(phi).abdev; }, 
    //                                             static_cast<real>(idx_min)*dphi - dphi, 
    //                                             static_cast<real>(idx_min)*dphi + dphi) };
    // real phi1 { min1_res.x };

    // // // Hone the Second Minimum
    // const MinResult1D min2_res { golden_section([&](real phi) { return departure_model(phi).abdev; },
    //                                             phi1+pi-dphi,
    //                                             phi1+pi+dphi) };
    // real phi2 { min2_res.x };

    // // Determine Which Phi is best 
    // phi1 = std::fmod(phi1, 2*pi);
    // phi2 = std::fmod(phi2, 2*pi);
    // std::pair<LinRegResult, LinRegResult> phi_fits { departure_model(phi1), departure_model(phi2) };
    // if ((phi_fits.first.b > 0) && (phi_fits.second.b > 0))
    //     throw InvalidArgument("No positive nu");
    
    // real phi {};
    // real nu {};
    // real omega_fsq {};
    // if (phi_fits.first.b < phi_fits.second.b)
    // {
    //     phi = phi1;
    //     nu = -phi_fits.first.b;
    //     omega_fsq = phi_fits.first.a;
    // } else 
    // {
    //     phi = phi2;
    //     nu = -phi_fits.second.b;
    //     omega_fsq = phi_fits.second.a;
    // }


    // // FINAL PARAMETERS
    // std::cout << "\n\n\n=======================";
    // std::cout << std::format(
    //     "Final Parameters\n"
    //     "a = {}\n"
    //     "b = {}\n"
    //     "phi = {}\n"
    //     "nu = {}\n"
    //     "omega_fsq = {}\n",
    // a, b, phi, nu, omega_fsq);


    // return 1;
}