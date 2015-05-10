#ifndef _CUDA_ALGEBRA_H_
#define _CUDA_ALGEBRA_H_

#include <vector>

#include "matrix.h"

using std::vector;

template <typename T>
T cuda_dot_product(const vector<T> &v1, const vector<T> &v2);

template <typename T>
void cuda_multiply_inplace(vector<T> &v, const T &k);

template <typename T>
void cuda_add_inplace(vector<T> &v, const T &k);

template <typename T>
void cuda_saxpy_inplace(vector<T> &y, const T &a, const vector<T> &x);

template <typename T>
vector<T> cuda_naive_multiply(const csr_matrix<T> &m, const vector<T> &v);

template <typename T>
vector<T> cuda_warp_multiply(const csr_matrix<T> &m, const vector<T> &v);

template <typename T>
vector<T> cuda_new_multiply(const csr_matrix<T> &m, const vector<T> &v);

template <typename T>
T cuda_l2_norm(const vector<T> &v);

template <typename T>
vector<T> cuda_lanczos_eigen(const csr_matrix<T> &matrix,
    int k, int steps);

#endif
