#pragma once

#include <cmath>
#include <utility>
#include <span>
#include <vector>

#include <linalg/Matrix.h>

#include <stator/core/constants.h>
#include <stator/core/concepts.h>
#include <stator/core/errors.h>
#include <stator/core/types.h>

namespace stator::core {

namespace detail {
// Implementation Details for Levenberg-Marquardt Fitting

// Covariance decompression
// Contract: Caller Ensures that covars.rows() == ma, covars.cols() == ma, and ia.size == ma()
//           and sum(ia) = mfit
inline void covsrt(linalg::Matrix<real>& covar, const std::vector<bool>& ia)
{
    // Get Sizes
    const idx ma { covar.rows() };
    const idx mfit { static_cast<idx>(std::accumulate(ia.begin(), ia.end(), 0)) };

    // Compiled out checks
    assert(ia.size() == ma);
    assert(mfit <= ma);
    assert((covar.rows()==ma)==(covar.cols()==ma));

    // Early Return
    if (ma == mfit) return; // Don't waste iterations (improvement on NR)

    // Zeros outside the mfit block
    for (idx i {mfit}; i < ma; i++)
        for (idx j {}; j < i +1; j++) covar[i,j] = covar[j,i] = 0.0;
    idx k { mfit - 1 };

    // Decompress
    for (idx j {ma}; j > 0; j--) // Deviation from NR, idx typdef is unsigned
    {
        if (ia[j-1]) 
        {
            std::swap_ranges(covar.colit(k).begin(), covar.colit(k).end(), covar.colit(j-1).begin()); 
            std::swap_ranges(covar.rowit(k).begin(), covar.rowit(k).end(), covar.rowit(j-1).begin()); 
            k--;
        }
    }
}


// Contract: Caller ensures that:
//           alpha.rows() == mfit, alpha.cols() == mfit, beta.size() == mfit
//           a.size() == ma, ia.size == ma()
//           and sum(ia) = mfit, x/y/z.size() all equal
//           sig[all] is > 0
template <ArgsValJac F>
void mrqcof(F&& func,
            const std::vector<real>& x,
            const std::vector<real>& y,
            const std::vector<real>& sig,
            const std::vector<real>& a,
            linalg::Matrix<real>& alpha,
            std::vector<real>& beta,
            const std::vector<bool>& ia,
            real& chisq)
{
    // Derive Constants
    const idx ndat { x.size() };
    const idx mfit { alpha.rows() };
    const idx ma { a.size() };

    // Compile out checks
    assert(
        (alpha.rows() == mfit) &&
        (alpha.cols() == mfit) &&
        (beta.size() == mfit)
    );
    assert(mfit <= ma);
    assert(x.size() == ndat);
    assert(y.size() == ndat);
    assert(sig.size() == ndat);
    assert(static_cast<idx>(std::accumulate(ia.begin(), ia.end(), 0)) == mfit);

    // Setup
    idx i {}, j {}, k {}, l {}, m {};
    real ymod {}, wt {}, sig2i {}, dy {};
    std::vector<real> dyda(ma, 0.0);

    // Initialize alpha, beta
    for (j = 0; j < mfit; j++)
    {
        for (k = 0; k <= j; k++) alpha[j, k] = 0.0;
        beta[j] = 0;
    }

    // Main Loop
    chisq = 0.0;
    for (i = 0; i < ndat; i++)
    {
        ymod = func(x[i], a, dyda); // Evaluate f(x, a) and Jacobian
        sig2i = 1.0 / (std::pow(sig[i], 2));
        dy = y[i] - ymod;
        for (j = 0, l = 0; l < ma; l++) {
            if (ia[l]) {
                wt = dyda[l] * sig2i;
                for (k = 0, m = 0; m < l + 1; m++)
                    if (ia[m]) alpha[j, k++] += wt * dyda[m];
                beta[j++] += dy * wt;
            }
        }
        chisq += dy * dy * sig2i;
    }
    for (j = 1; j < mfit; j++)
        for (k = 0; k < j; k++) alpha[k, j] = alpha[j, k];
}

}

}