#ifndef _EIGEN_H_
#define _EIGEN_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "linear_algebra.h"
#include "matrix.h"

using std::vector;

/**
 * @brief   Lanczos algorithm for eigendecomposition.
 * 
 * @param   matrix  CSR matrix to decompose
 * @param   k       number of largest eigenvalues to compute
 * @param   steps   maximum steps for the iteration
 * @tparam  T       matrix element data type
 * @return  list of eigenvalues
 */
template <typename T>
vector<T> lanczos_eigen(const csr_matrix<T> &matrix, int k, int steps) {
    int n = matrix.row_size();
    assert(n > 0 && n == matrix.col_size());
    assert(steps > 2 * k);

    symm_tridiag_matrix<T> tridiag(steps);
    vector<vector<T> > basis;

    vector<T> r(n, 0);
    r[0] = 1; // initialize a "random" vector
    T beta = l2_norm(r);
    for (int t = 0; t < steps; ++t) {
        if (t > 0) {
            tridiag.beta(t - 1) = beta;
        }
        multiply_inplace(r, 1 / beta);
        basis.push_back(r);
        r = multiply(matrix, r);
        T alpha = dot_product(basis[t], r);
        saxpy_inplace(r, -alpha, basis[t]);
        if (t > 0) {
            saxpy_inplace(r, -beta, basis[t - 1]);
        }
        tridiag.alpha(t) = alpha;
        beta = l2_norm(r);
    }
    return lanczos_no_spurious(tridiag, k);
}

template <typename T>
vector<T> lanczos_no_spurious(symm_tridiag_matrix<T> &tridiag, int k, const T epsilon = 1e-3) {
    assert(tridiag.size() > 0);
    vector<T> eigen = qr_eigen(tridiag);
    sort(eigen.rbegin(), eigen.rend());
    tridiag.remove_forward(0);
    vector<T> test_eigen = qr_eigen(tridiag);
    vector<T> result;

    int i = 0;
    int j = 0;
    while (j <= eigen.size()) { // scan through one position beyond the end of the list
        if (j < eigen.size() && std::abs(eigen[j] - eigen[i]) < epsilon) {
            j++;
            continue;
        }
        // simple eigenvalues not in test set are preserved
        // multiple eigenvalues are only preserved once
        if (j - i > 1 || approximate_find(test_eigen, eigen[i], epsilon) == test_eigen.end()) {
            result.push_back(eigen[i]);
        }
        i = j++;
    }
    std::sort(result.rbegin(), result.rend());
    result.resize(std::min((int)result.size(), k));
    return result;
}

/**
 * @brief   QR eigendecomposition for symmetric tridiagonal matrices.
 * 
 * @param   matrix  symmetric tridiagonal matrix to decompose
 * @param   epsilon precision threshold
 * @tparam  T       matrix element data type
 * @return  list of eigenvalues
 */
template <typename T>
vector<T> qr_eigen(const symm_tridiag_matrix<T> &matrix, const T epsilon = 1e-8) {
    symm_tridiag_matrix<T> tridiag = matrix;
    int n = tridiag.size();

    tridiag.resize(n + 1);
    tridiag.alpha(n) = 0;
    tridiag.beta(n - 1) = 0;
    for (int i = 0; i < n - 1; ++i) {
        tridiag.beta(i) = tridiag.beta(i) * tridiag.beta(i);
    }
    bool converged = false;
    while (!converged) {
        T diff(0);
        T u(0);
        T ss2(0), s2(0); // previous and current value of s^2
        for (int i = 0; i < n; ++i) {
            T gamma = tridiag.alpha(i) - u;
            T p2 = T(std::abs(1 - s2)) < epsilon ? (1 - ss2) * tridiag.beta(i - 1) : gamma * gamma / (1 - s2);
            if (i > 0) {
                tridiag.beta(i - 1) = s2 * (p2 + tridiag.beta(i));
            }
            ss2 = s2;
            s2 = tridiag.beta(i) / (p2 + tridiag.beta(i));
            u = s2 * (gamma + tridiag.alpha(i + 1));
            // update alpha
            T old = tridiag.alpha(i);
            tridiag.alpha(i) = gamma + u;
            diff = std::max(diff, T(std::abs(old - tridiag.alpha(i))));
        }
        if (diff < epsilon) {
            converged = true;
        }
    }
    return vector<T>(tridiag.alpha_data(), tridiag.alpha_data() + n);
}

#endif
