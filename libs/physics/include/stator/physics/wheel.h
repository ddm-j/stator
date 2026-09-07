#pragma once

#include <vector>
#include <cmath>

#include <linalg/solvers.h>

#include <stator/core/types.h>

using namespace stator::core;

namespace stator::physics {

inline std::pair<real, real> fit_rotor(const std::vector<std::vector<real>>& ts)
{
    // A super memory inneficient matrix solve to fit rotor deceleration
    // but uses my linalg library!

    using linalg::Matrix;

    // Index Constants
    const idx num_spins { ts.size() };
    const idx width { num_spins + 1 };
    const idx k_col { num_spins };
    idx num_rows {};
    for (idx i {}; i < ts.size(); i++)
        num_rows += ts[i].size(); 

    // Create Matrix A and Vector y
    Matrix<real> A(num_rows, width);
    Matrix<real> y(num_rows, 1);
    idx row {};
    for (idx spin {}; spin < num_spins; spin++)
        for (idx sample {}; sample < ts[spin].size(); sample++)
        {
            const real t { ts[spin][sample] - ts[spin][0] };
            if (t == 0.0) continue;

            A[row, spin] = t;
            A[row, k_col] = -t*t/2.0;
            y[row, 0] = static_cast<real>(sample);
            ++row;
        }

    // Least Squares Solve
    Matrix<real> At = Matrix<real>(A.transpose());
    Matrix<real> AtA = At * A;
    Matrix<real> Aty = At * y;
    Matrix<real> p { linalg::solvers::solve_cholesky(AtA, Aty) };

    // The rotor decel constant 
    const real k { p[k_col, 0] };

    // Measure k uncertainty
    Matrix<real> r { y - A*p };
    const real sig2 { (Matrix<real>(r.transpose()) * r)[0,0] / static_cast<real>(row-width) };
    Matrix<real> e_k(width, 1);
    e_k[k_col, 0] = 1.0;
    Matrix<real> z { linalg::solvers::solve_cholesky(AtA, e_k) };

    const real sigma_k { sqrt(sig2 * z[k_col, 0]) };
    return {k, sigma_k/k};
};

}