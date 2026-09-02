#pragma once

#include <cmath>
#include <utility>
#include <span>
#include <vector>
#include <cassert>

#include <linalg/Matrix.h>
#include <linalg/solvers.h>

#include <stator/core/constants.h>
#include <stator/core/concepts.h>
#include <stator/core/errors.h>
#include <stator/core/types.h>
#include <stator/core/numeric_result.h>
#include <stator/core/numerical_jacobian.h>

namespace stator::core {

namespace detail {
// Implementation Details for Levenberg-Marquardt Fitting

// Utilities
inline bool not_all_finite(const std::vector<real> vec)
{
    const auto is_nonfinite = [](real val) -> bool {
        return !std::isfinite(val);
    };
    return std::any_of(vec.begin(), vec.end(), is_nonfinite);
}

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
    const idx ma { a.size() };
    const idx mfit { static_cast<idx>(std::count(ia.begin(), ia.end(), true)) };

    // Compile out checks
    assert(y.size()   == ndat);
    assert(sig.size() == ndat);
    assert(ia.size()  == ma);
    assert(mfit > 0);
    assert(alpha.rows() >= mfit && alpha.cols() >= mfit);
    assert(beta.size()  >= mfit);

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

// Overload: User Supplies ArgsValJac (analytical or custom jacobian)
template <ArgsValJac F>
FitResult fitmrq(
    F&& func,
    const std::vector<real>& x,
    const std::vector<real>& y,
    const std::vector<real>& sig,
    std::vector<real> a,            // Parameter Vector is taken by copy
    const std::vector<bool>& ia,
    real tol = 0.0,
    const idx MAXIT = 1000
)
{
    // Argument Checking
    // // Input Shapes
    if ((x.size() != y.size()) || (y.size() != sig.size()))
        throw InvalidArgument("fitmrq(): User provided x, y, sig data do not match in size.");
    if (ia.size() != a.size())
        throw InvalidArgument("fitmrq(): User provided boolean mask (ia) does not match parameters (a) in size.");
    if (static_cast<idx>(std::accumulate(ia.begin(), ia.end(), 0)) == 0)
        throw InvalidArgument("fitmrq(): User provided boolean mask (ia) cannot be all false.");
    // // Finiteness (open question: should I allow non-finite parameters (a)?)
    if (
        detail::not_all_finite(x)   ||
        detail::not_all_finite(y)   ||
        detail::not_all_finite(sig) ||
        detail::not_all_finite(a)
    )
        throw InvalidArgument("fitmrq(): User provided x, y, sig, or params (a) contain non-finite values.");
    // // Any zero or negative sig
    if (std::any_of(sig.begin(), sig.end(), [](real val){ return val <= 0.0; }))
        throw InvalidArgument("fitmrq(): User provided values for sig must be greater than zero.");

    // // Tolerance
    constexpr real TOL_MIN = 16.0 * real_EPS;
    constexpr real TOL_DEFAULT = 1e-10;
    if (tol <= 0.0)
        tol = TOL_DEFAULT;
    else if (tol < TOL_MIN)
        tol = TOL_MIN;

    // Constexprs
    constexpr idx NDONE { 4 };

    // Allocation
    // // Sizes
    const idx ma { a.size() };
    const idx mfit { static_cast<idx>(std::accumulate(ia.begin(), ia.end(), 0)) };
    // // Indexes
    idx j {}, k {}, l {}, iter {}, done {};
    // // Real Values
    real alambda { 0.001 };
    real chisq {}, ochisq {};
    // // Vectors
    std::vector<real> atry { a };
    std::vector<real> beta(ma, 0.0);
    std::vector<real> da(ma, 0.0);
    // // Matrices
    linalg::Matrix<real> alpha(ma, ma);
    linalg::Matrix<real> covar(ma, ma);
    linalg::Matrix<real> temp(mfit, mfit);
    linalg::Matrix<real> oneda(mfit, 1);

    // Initialization
    detail::mrqcof(func, x, y, sig, a, alpha, beta, ia, chisq);
    ochisq = chisq;

    // Main Loop
    for (iter = 0; iter < MAXIT; iter++)
    {
        // Last Pass Flag Detection
        if (done == NDONE) alambda = 0.0;

        // Alter alpha's diagonal with alambda
        for (j = 0; j < mfit; j++)
        {
            for (k = 0; k < mfit; k++) covar[j, k] = alpha[j, k];
            covar[j, j] = alpha[j, j] * (1.0 + alambda);
            for (k = 0; k < mfit; k++) temp[j, k] = covar[j, k];
            oneda[j, 0] = beta[j];
        }

        // Gauss-Jordan Elimination
        linalg::solvers::solve_gauss_jordan(temp, oneda);
        for (j = 0; j < mfit; j++)
        {
            for (k = 0; k < mfit; k++) covar[j, k] = temp[j, k];
            da[j] = oneda[j, 0];
        }

        // Return if Last Pass
        if (done == NDONE)
        {
            detail::covsrt(covar, ia);
            detail::covsrt(alpha, ia);
            FitResult res { iter, true, a, covar, beta, chisq };
            return res;
        }

        // Trial Success?
        for (j = 0, l = 0; l < ma; l++)
            if (ia[l]) atry[l] = a[l] + da[j++];
        detail::mrqcof(func, x, y, sig, atry, covar, da, ia, chisq);
        if (std::abs(chisq - ochisq) < std::max(tol, tol*chisq)) done++;
        if (chisq < ochisq)
        { // Trial Succeeded: Decrease alambda
            alambda *= 0.1;
            ochisq = chisq;
            for (j = 0; j < mfit; j++) {
                for (k = 0; k < mfit; k++) alpha[j, k] = covar[j, k];
                beta[j] = da[j];
            }
            // Update the Parameter Vector
            for (l = 0; l < ma; l++) a[l] = atry[l];
        } else
        { // Trial Failed: Increase alambda
            alambda *= 10.0;
            chisq = ochisq;
        }
    }
    throw ConvergenceFailure("fitmrq(): Unable to converge in {} iterations with {}/4 relaxations.", MAXIT, done);
}

// Overload: User Supplies ArgsVal (builtin numerical jacobian is used)
template <ArgsVal F>
FitResult fitmrq(
    F&& func,
    const std::vector<real>& x,
    const std::vector<real>& y,
    const std::vector<real>& sig,
    std::vector<real> a,            // Parameter Vector is taken by copy
    const std::vector<bool>& ia,
    real tol = 0.0,
    const idx MAXIT = 1000
)
{
    return fitmrq(NumericalJacobian(func, a.size()), x, y, sig, a, ia, tol, MAXIT);
}

// Overload: User supplies ArgsVal, x, y, (scalar) sig, and a
template <ArgsVal F>
FitResult fitmrq(
    F&& func,
    const std::vector<real>& x,
    const std::vector<real>& y,
    real sig_scalar,
    std::vector<real> a,            // Parameter Vector is taken by copy
    real tol = 0.0,
    const idx MAXIT = 1000
)
{
    // Create Sig & ia
    const std::vector<real> sig(x.size(), sig_scalar);
    const std::vector<bool> ia(a.size(), true);

    return fitmrq(NumericalJacobian(func, a.size()), x, y, sig, a, ia, tol, MAXIT);
}

}