#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>

#include "matrix.h"
#include "cuda_algebra.h"

#define WARP_SIZE 32
#define THREADS_PER_BLOCK 512

using std::vector;

/**
 * @brief   Cuda kernel function for vector dot product.
 *
 * @param   N   The vector size.
 * @param   x   The first input vector.
 * @param   y   The second input vector.
 * @param   z   The temp sum per block.
 */
template <typename T>
__global__ void dot_product_kernel(const int N, const T *x, const T *y,
    T *z) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ T result[THREADS_PER_BLOCK];

    if (index < N) {
        result[threadIdx.x] = x[index] * y[index];
    } else {
        result[threadIdx.x] = 0;
    }
    __syncthreads();

    int half = THREADS_PER_BLOCK / 2;
    while (half > 0) {
        if (threadIdx.x < half) {
            result[threadIdx.x] += result[threadIdx.x + half];
        }
        __syncthreads();
        half /= 2;
    }

    if (threadIdx.x == 0) {
        z[blockIdx.x] = result[0];
    }
}

/**
 * @brief   Cuda kernel function for vector multiply in place.
 *
 * @param   N   The vector size.
 * @param   x   The input vector.
 * @param   k   The value to multiply.
 */
template <typename T>
__global__ void multiply_inplace_kernel(const int N, T *x, const T k) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        x[index] = x[index] * k;
    }
}

/**
 * @brief   Cuda kernel function for vector add in place.
 *
 * @param   N   The vector size.
 * @param   x   The input vector.
 * @param   k   The value to add.
 */
template <typename T>
__global__ void add_inplace_kernel(const int N, T *x, const T k) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        x[index] = x[index] + k;
    }
}

/**
 * @brief   Cuda kernel function for vector saxpy in place(y += a * x).
 *
 * @param   N   The vector size.
 * @param   y   The output vector.
 * @param   x   The input vector.
 * @param   a   The value to multiply.
 */
template <typename T>
__global__ void saxpy_inplace_kernel(const int N, T *y, const T *x,
    const T a) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        y[index] += a * x[index];
    }
}

/**
 * @brief   Cuda kernel function for naive sparse matrix multiplication.
 *
 * @param   rows    The row number of the matrix.
 * @param   row_ptr Row pointers in the CSR matrix.
 * @param   col_ind Column indexes in the CSR matrix.
 * @param   values  Data values in the CSR matrix.
 * @param   x       The input vector x to multiply.
 * @param   y       The output vector y.
 */
template <typename T>
__global__ void naive_multiply_kernel(const int rows, const int *row_ptr,
    const int *col_ind, const T *values, const T *x, T *y) {

    int r = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows) {
        T result(0);
        int start = row_ptr[r];
        int end = row_ptr[r + 1];
        for (int i = start; i < end; i++) {
            result += values[i] * x[col_ind[i]];
        }
        y[r] = result;
    }
}

/**
 * @brief   Cuda kernel function for naive sparse matrix multiplication.
 *
 * @param   rows    The row number of the matrix.
 * @param   row_ptr Row pointers in the CSR matrix.
 * @param   col_ind Column indexes in the CSR matrix.
 * @param   values  Data values in the CSR matrix.
 * @param   x       The input vector x to multiply.
 * @param   y       The output vector y.
 */
template <typename T>
__global__ void warp_multiply_kernel(const int rows, const int *row_ptr,
    const int *col_ind, const T *values, const T *x, T *y) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int r = index / WARP_SIZE;
    int lane = index % WARP_SIZE;
    __shared__ T result[THREADS_PER_BLOCK];

    if (r < rows) {
        int start = row_ptr[r];
        int end = row_ptr[r + 1];
        result[threadIdx.x] = 0;
        for (int i = start + lane; i < end; i+= WARP_SIZE) {
            result[threadIdx.x] += values[i] * x[col_ind[i]];
        }
        // Threads in a warp are synchronized, so we can do this
        int half = WARP_SIZE / 2;
        while (half > 0) {
            if (lane < half) {
                result[threadIdx.x] += result[threadIdx.x + half];
            }
            half /= 2;
        }
        if (lane == 0) {
            y[r] = result[threadIdx.x];
        }
    }
}

/**
 * @brief   Caller function for vector dot product in CUDA.
 *
 * @param   v1  The first vector.
 * @param   v2  The second vector.
 *
 * @return  The result of dot product of v1 and v2.
 */
template <typename T>
T cuda_dot_product(const vector<T> &v1, const vector<T> &v2) {
    int n = v1.size();
    assert(n == v2.size());
    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Malloc device space
    T *x, *y, *z;
    cudaMalloc(&x, sizeof(T) * n);
    cudaMalloc(&y, sizeof(T) * n);
    cudaMalloc(&z, sizeof(T) * blocks);

    // Transfer data from host to device
    cudaMemcpy(x, v1.data(), sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y, v2.data(), sizeof(T) * n, cudaMemcpyHostToDevice);

    // Run kernel
    dot_product_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(n, x, y, z);
    cudaThreadSynchronize();

    // Transfer result back from device to host
    T z_host[blocks];
    T result(0);
    cudaMemcpy(z_host, z, sizeof(T) * blocks, cudaMemcpyDeviceToHost);
    for (int i = 0; i < blocks; i++) {
        result += z_host[i];
    }

    // Release device space
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return result;
}

/**
 * @brief   Caller function for inplace vector multiplication in CUDA.
 *
 * @param   v   The vector to multiply to.
 * @param   k   The value to multiply.
 */
template <typename T>
void cuda_multiply_inplace(vector<T> &v, const T &k) {
    int n = v.size();

    // Malloc device space
    T *x;
    cudaMalloc(&x, sizeof(T) * n);

    // Transfer data from host to device
    cudaMemcpy(x, v.data(), sizeof(T) * n, cudaMemcpyHostToDevice);

    // Run kernel
    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    multiply_inplace_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(n, x, k);
    cudaThreadSynchronize();

    // Transfer result back from device to host
    cudaMemcpy(v.data(), x, sizeof(T) * n, cudaMemcpyDeviceToHost);

    // Release device space
    cudaFree(x);
}

/**
 * @brief   Caller function for inplace vector add in CUDA.
 *
 * @param   v   The vector to add to.
 * @param   k   The value to add.
 */
template <typename T>
void cuda_add_inplace(vector<T> &v, const T &k) {
    int n = v.size();

    // Malloc device space
    T *x;
    cudaMalloc(&x, sizeof(T) * n);

    // Transfer data from host to device
    cudaMemcpy(x, v.data(), sizeof(T) * n, cudaMemcpyHostToDevice);

    // Run kernel
    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    add_inplace_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(n, x, k);
    cudaThreadSynchronize();

    // Transfer result back from device to host
    cudaMemcpy(v.data(), x, sizeof(T) * n, cudaMemcpyDeviceToHost);

    // Release device space
    cudaFree(x);
}

/**
 * @brief   Caller function for inplace vector saxpy (y += a * x) in CUDA.
 *
 * @param   y   The vector to add to.
 * @param   x   The vector to multiply to.
 * @param   a   The value to multiply.
 */
template <typename T>
void cuda_saxpy_inplace(vector<T> &y, const T &a, const vector<T> &x) {
    int n = y.size();
    assert(n = x.size());

    // Malloc device space
    T *dev_y;
    T *dev_x;
    cudaMalloc(&dev_y, sizeof(T) * n);
    cudaMalloc(&dev_x, sizeof(T) * n);

    // Transfer data from host to device
    cudaMemcpy(dev_y, y.data(), sizeof(T) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, x.data(), sizeof(T) * n, cudaMemcpyHostToDevice);

    // Run kernel
    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    saxpy_inplace_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(n, dev_y, dev_x,
        a);
    cudaThreadSynchronize();

    // Transfer result back from device to host
    cudaMemcpy(y.data(), dev_y, sizeof(T) * n, cudaMemcpyDeviceToHost);

    // Release device space
    cudaFree(dev_y);
    cudaFree(dev_x);
}

/**
 * @brief   Caller function for naive sparse matrix multiplication in CUDA.
 *
 * @param   m   The matrix to multiply.
 * @param   v   The vector to multiply.
 *
 * @return  The result of matrix vector multiplication of m*v.
 */
template <typename T>
vector<T> cuda_naive_multiply(const csr_matrix<T> &m, const vector<T> &v) {
    int rows = m.row_size();
    int cols = m.col_size();
    int nonzeros = m.nonzeros();
    assert(cols == v.size());

    // Malloc device space
    int *row_ptr, *col_ind;
    T *values, *x, *y;
    cudaMalloc(&row_ptr, sizeof(int) * (rows + 1));
    cudaMalloc(&col_ind, sizeof(int) * nonzeros);
    cudaMalloc(&values, sizeof(T) * nonzeros);
    cudaMalloc(&x, sizeof(T) * cols);
    cudaMalloc(&y, sizeof(T) * cols);

    // Transfer data from host to device
    cudaMemcpy(row_ptr, m.row_ptr_data(), sizeof(int) * (rows + 1),
        cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind, m.col_ind_data(), sizeof(int) * nonzeros,
        cudaMemcpyHostToDevice);
    cudaMemcpy(values, m.values_data(), sizeof(T) * nonzeros,
        cudaMemcpyHostToDevice);
    cudaMemcpy(x, v.data(), sizeof(T) * cols, cudaMemcpyHostToDevice);

    // Run kernel
    const int blocks = (rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    naive_multiply_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(rows, row_ptr,
        col_ind, values, x, y);
    cudaThreadSynchronize();

    // Transfer result back from device to host
    vector<T> result(cols);
    cudaMemcpy(result.data(), y, sizeof(T) * cols, cudaMemcpyDeviceToHost);

    // Release device space
    cudaFree(row_ptr);
    cudaFree(col_ind);
    cudaFree(values);
    cudaFree(x);
    cudaFree(y);

    return result;
}

/**
 * @brief   Caller function for warp sparse matrix multiplication in CUDA.
 *
 * @param   m   The matrix to multiply.
 * @param   v   The vector to multiply.
 *
 * @return  The result of matrix vector multiplication of m*v.
 */
template <typename T>
vector<T> cuda_warp_multiply(const csr_matrix<T> &m, const vector<T> &v) {
    int rows = m.row_size();
    int cols = m.col_size();
    int nonzeros = m.nonzeros();
    assert(cols == v.size());

    // Malloc device space
    int *row_ptr, *col_ind;
    T *values, *x, *y;
    cudaMalloc(&row_ptr, sizeof(int) * (rows + 1));
    cudaMalloc(&col_ind, sizeof(int) * nonzeros);
    cudaMalloc(&values, sizeof(T) * nonzeros);
    cudaMalloc(&x, sizeof(T) * cols);
    cudaMalloc(&y, sizeof(T) * cols);

    // Transfer data from host to device
    cudaMemcpy(row_ptr, m.row_ptr_data(), sizeof(int) * (rows + 1),
        cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind, m.col_ind_data(), sizeof(int) * nonzeros,
        cudaMemcpyHostToDevice);
    cudaMemcpy(values, m.values_data(), sizeof(T) * nonzeros,
        cudaMemcpyHostToDevice);
    cudaMemcpy(x, v.data(), sizeof(T) * cols, cudaMemcpyHostToDevice);

    // Run kernel
    const int warps_per_block = THREADS_PER_BLOCK / WARP_SIZE;
    const int blocks = (rows + warps_per_block - 1) / warps_per_block;
    warp_multiply_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(rows, row_ptr,
        col_ind, values, x, y);
    cudaThreadSynchronize();

    // Transfer result back from device to host
    vector<T> result(cols);
    cudaMemcpy(result.data(), y, sizeof(T) * cols, cudaMemcpyDeviceToHost);

    // Release device space
    cudaFree(row_ptr);
    cudaFree(col_ind);
    cudaFree(values);
    cudaFree(x);
    cudaFree(y);

    return result;
}

template __global__ void dot_product_kernel<float>(const int,
    const float *, const float *, float *);
template float cuda_dot_product<float>(const vector<float> &v1,
    const vector<float> &v2);
template __global__ void multiply_inplace_kernel<float>(const int, float *,
    const float);
template void cuda_multiply_inplace<float>(vector<float> &v, const float &k);
template __global__ void add_inplace_kernel<float>(const int, float *,
    const float);
template void cuda_add_inplace<float>(vector<float> &v, const float &k);
template __global__ void saxpy_inplace_kernel<float>(const int, float *,
    const float *, const float);
template void cuda_saxpy_inplace<float>(vector<float> &y, const float &a,
    const vector<float> &x);

template __global__ void naive_multiply_kernel(const int, const int *,
    const int *, const float *, const float *, float *);
template vector<float> cuda_naive_multiply(const csr_matrix<float> &m,
    const vector<float> &v);
template __global__ void warp_multiply_kernel(const int, const int *,
    const int *, const float *, const float *, float *);
template vector<float> cuda_warp_multiply(const csr_matrix<float> &m,
    const vector<float> &v);
