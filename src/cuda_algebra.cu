#include <cassert>
#include <cuda.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include <cmath>
#include <utility>

#include "eigen.h"
#include "matrix.h"
#include "cuda_algebra.h"
#include "cycle_timer.h"

#define THREADS_PER_BLOCK 256

using std::vector;

/**
 * @brief   Cuda kernel function for vector copy.
 *
 * @param   N   The vector size.
 * @param   y   The dest vector.
 * @param   x   The src vector.
 */
template <typename T>
__global__ void vector_copy_kernel(const int N, T *y, const T *x) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        y[index] = x[index];
    }
}

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
 * @brief   Cuda kernel function for vector-vector add in place.
 *
 * @param   N   The vector size.
 * @param   x   The input vector.
 * @param   y   The other vector to add.
 */
template <typename T>
__global__ void vec_add_inplace_kernel(const int N, T *x, const T *y) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        x[index] = x[index] + y[index];
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
 * @brief   Cuda kernel function for warp sparse matrix multiplication.
 *
 * @param   group_size  The number of threads used to calculate one row.
 * @param   rows        The row number of the matrix.
 * @param   begin_row   The row to begin from in this kernel launch.
 * @param   row_ptr     Row pointers in the CSR matrix.
 * @param   col_ind     Column indexes in the CSR matrix.
 * @param   values      Data values in the CSR matrix.
 * @param   x           The input vector x to multiply.
 * @param   y           The output vector y.
 */
template <typename T>
__global__ void warp_multiply_kernel(const int group_size, const int rows,
    const int begin_row, const int *row_ptr, const int *col_ind,
    const T *values, const T *x, T *y) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int r = index / group_size + begin_row;
    int lane = index % group_size;
    __shared__ volatile T result[THREADS_PER_BLOCK];

    result[threadIdx.x] = 0;
    if (r < rows) {
        int start = row_ptr[r];
        int end = row_ptr[r + 1];
        for (int i = start + lane; i < end; i+= group_size) {
            result[threadIdx.x] += values[i] * x[col_ind[i]];
        }
        // Threads in a warp are synchronized, so we can do this
        int half = group_size / 2;
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
 * @brief   Cuda kernel function for new sparse matrix multiplication.
 *
 * @param   rows    The row number of the matrix.
 * @param   row_ptr Row pointers in the CSR matrix.
 * @param   col_ind Column indexes in the CSR matrix.
 * @param   row_ind Row indexes in the CSR matrix.
 * @param   values  Data values in the CSR matrix.
 * @param   x       The input vector x to multiply.
 * @param   y       The output vector y.
 */
template <typename T>
__global__ void new_multiply_kernel(const int rows, const int *row_ptr,
    const int *col_ind, const int *row_ind, const T *values, const T *x,
    T *y) {

    const int group_size = 32;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = index % group_size;
    __shared__ volatile T result[THREADS_PER_BLOCK][group_size];

    int values_len = row_ptr[rows];
    int max_warp_index = index -lane + group_size - 1;
    int max_row;
    if (max_warp_index < values_len) {
        max_row = row_ind[max_warp_index] % group_size;
    } else {
        max_row = (rows - 1) % group_size;
    }
    int min_warp_index = index - lane;
    int min_row = row_ind[min_warp_index] % group_size;
    if (min_row < max_row) {
        for (int i = min_row; i <= max_row; i++) {
            result[threadIdx.x][i] = 0;
        }
    } else {
        for (int i = max_row; i < group_size; i++) {
            result[threadIdx.x][i] = 0;
        }
        for (int i = 0; i <= min_row; i++) {
            result[threadIdx.x][i] = 0;
        }
    }

    if (index < values_len) {
        int row_id = row_ind[index];
        result[threadIdx.x][row_id % group_size] +=
            values[index] * x[col_ind[index]];
        // Threads in a warp are synchronized, so we can do this
        int half = group_size / 2;
        while (half > 0) {
            if (lane < half) {
                if (min_row < max_row) {
                    for (int i = min_row; i <= max_row; i++) {
                        result[threadIdx.x][i] += result[threadIdx.x+half][i];
                    }
                } else {
                    for (int i = max_row; i < group_size; i++) {
                        result[threadIdx.x][i] += result[threadIdx.x+half][i];
                    }
                    for (int i = 0; i <= min_row; i++) {
                        result[threadIdx.x][i] += result[threadIdx.x+half][i];
                    }
                }
            }
            half /= 2;
        }

        if (lane == 0 || row_id > row_ind[index - 1]) {
            atomicAdd(&y[row_id],
                result[threadIdx.x - lane][row_id % group_size]);
        }
    }
}

template <typename T>
T device_dot_product(int n, const T *device_x, const T *device_y, T *device_scratch) {
    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Run kernel
    dot_product_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(
        n, device_x, device_y, device_scratch);
    cudaThreadSynchronize();

    // Transfer result back from device to host
    T host_scratch[blocks];
    T result(0);
    cudaMemcpy(host_scratch, device_scratch, sizeof(T) * blocks, cudaMemcpyDeviceToHost);
    for (int i = 0; i < blocks; i++) {
        result += host_scratch[i];
    }
    return result;
}

/**
 * @brief   Cuda kernel function for dynamic sparse matrix multiplication.
 *
 * @param   rows    The row number of the matrix.
 * @param   row_ptr Row pointers in the CSR matrix.
 * @param   col_ind Column indexes in the CSR matrix.
 * @param   values  Data values in the CSR matrix.
 * @param   x       The input vector x to multiply.
 * @param   y       The output vector y.
 */
template <typename T>
__global__ void dynamic_multiply_kernel(const int rows,
    const int rows_per_thread, const int *row_ptr, const int *col_ind,
    const T *values, const T *x, T *y) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int begin_row = index * rows_per_thread;
    int end_row = (index + 1) * rows_per_thread - 1;

    if (begin_row < rows) {
        end_row = end_row < rows ? end_row : rows - 1;
        int nonzeros = row_ptr[end_row+1] - row_ptr[begin_row];
        int row_nonzeros = nonzeros / rows_per_thread;
        int group_size = row_nonzeros > 16 ? 32 : 16;
        group_size = row_nonzeros > 8 ? group_size : 8;
        group_size = row_nonzeros > 4 ? group_size : 4;
        group_size = row_nonzeros > 2 ? group_size : 2;
        const int groups_per_block = THREADS_PER_BLOCK / group_size;
        const int blocks = (rows_per_thread + groups_per_block - 1) / groups_per_block;
        warp_multiply_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(group_size, rows,
            begin_row, row_ptr, col_ind, values, x, y);
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

    T result = device_dot_product(n, x, y, z);

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
 */ template <typename T>
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
 * @brief   Caller function for vector l2 norm in CUDA.
 *
 * @param   v  The vector.
 *
 * @return  The l2 norm of the vector v.
 */
template <typename T>
T cuda_l2_norm(const vector<T> &v) {
    int n = v.size();
    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Malloc device space
    T *x, *z;
    cudaMalloc(&x, sizeof(T) * n);
    cudaMalloc(&z, sizeof(T) * blocks);

    // Transfer data from host to device
    cudaMemcpy(x, v.data(), sizeof(T) * n, cudaMemcpyHostToDevice);

    T result = device_dot_product(n, x, x, z);

    // Release device space
    cudaFree(x);
    cudaFree(z);

    return T(sqrt(result));
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
    double start_time = cycle_timer::current_seconds();
    naive_multiply_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(rows, row_ptr,
        col_ind, values, x, y);
    cudaThreadSynchronize();
    double end_time = cycle_timer::current_seconds();
    printf("gpu naive multiply kernel: %f\n", end_time - start_time);

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
    const int row_nonzeros = nonzeros / rows;
    int group_size = row_nonzeros > 16 ? 32 : 16;
    group_size = row_nonzeros > 8 ? group_size : 8;
    group_size = row_nonzeros > 4 ? group_size : 4;
    group_size = row_nonzeros > 2 ? group_size : 2;
    const int groups_per_block = THREADS_PER_BLOCK / group_size;
    const int blocks = (rows + groups_per_block - 1) / groups_per_block;
    double start_time = cycle_timer::current_seconds();
    warp_multiply_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(group_size, rows,
        0, row_ptr, col_ind, values, x, y);
    cudaThreadSynchronize();
    double end_time = cycle_timer::current_seconds();
    printf("gpu warp multiply kernel: %f\n", end_time - start_time);


    // Transfer result back from device to host
    //vector<T> result(cols);
    //cudaMemcpy(result.data(), y, sizeof(T) * cols, cudaMemcpyDeviceToHost);
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

template <typename T>
cusparseStatus_t cusparse_mv_multiply_wrapper(cusparseHandle_t handle,
    cusparseOperation_t transA, int m, int n, int nnz, const cusparseMatDescr_t descrA,
    const T *csrValA, const int *csrRowPtrA, const int *csrColIndA, const T *x, T *y);

template <>
cusparseStatus_t cusparse_mv_multiply_wrapper<float>(cusparseHandle_t handle,
    cusparseOperation_t transA, int m, int n, int nnz, const cusparseMatDescr_t descrA,
    const float *csrValA, const int *csrRowPtrA, const int *csrColIndA, const float *x, float *y) {
    float one = 1;
    float zero = 0;
    return cusparseScsrmv(handle, transA, m, n, nnz, &one, descrA,
        csrValA, csrRowPtrA, csrColIndA, x, &zero, y);
}

template <>
cusparseStatus_t cusparse_mv_multiply_wrapper<double>(cusparseHandle_t handle,
    cusparseOperation_t transA, int m, int n, int nnz, const cusparseMatDescr_t descrA,
    const double *csrValA, const int *csrRowPtrA, const int *csrColIndA, const double *x, double *y) {
    double one = 1;
    double zero = 0;
    return cusparseDcsrmv(handle, transA, m, n, nnz, &one, descrA,
        csrValA, csrRowPtrA, csrColIndA, x, &zero, y);
}

/**
 * @brief   Caller function for cusparse matrix multiplication in CUDA.
 *
 * @param   m   The matrix to multiply.
 * @param   v   The vector to multiply.
 *
 * @return  The result of matrix vector multiplication of m*v.
 */
template <typename T>
vector<T> cuda_cusparse_multiply(const csr_matrix<T> &m,
    const vector<T> &v) {
    cusparseHandle_t handle=0;
    cusparseMatDescr_t descr=0;

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
    cudaMemset(y, 0, sizeof(T) * cols);

    // Transfer data from host to device
    cudaMemcpy(row_ptr, m.row_ptr_data(), sizeof(int) * (rows + 1),
        cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind, m.col_ind_data(), sizeof(int) * nonzeros,
        cudaMemcpyHostToDevice);
    cudaMemcpy(values, m.values_data(), sizeof(T) * nonzeros,
        cudaMemcpyHostToDevice);
    cudaMemcpy(x, v.data(), sizeof(T) * cols, cudaMemcpyHostToDevice);

    double start_time= cycle_timer::current_seconds();
    cusparseCreate(&handle);
    double end_time= cycle_timer::current_seconds();
    printf("gpu cusparse handle initialize time: %f\n", end_time - start_time);
    cusparseCreateMatDescr(&descr); 
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    start_time= cycle_timer::current_seconds();
    cusparse_mv_multiply_wrapper<T>(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        rows, cols, nonzeros, descr, values, row_ptr, col_ind, x, y);
    cudaThreadSynchronize();
    end_time= cycle_timer::current_seconds();
    printf("gpu cusparse multiply kernel: %f\n", end_time - start_time);
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);

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
 * @brief   Caller function for dynamic sparse matrix multiplication in CUDA.
 *
 * @param   m   The matrix to multiply.
 * @param   v   The vector to multiply.
 *
 * @return  The result of matrix vector multiplication of m*v.
 */
template <typename T>
vector<T> cuda_dynamic_multiply(const csr_matrix<T> &m, const vector<T> &v) {
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
    const int rows_per_thread = 32;
    const int rows_per_block = rows_per_thread * THREADS_PER_BLOCK;
    const int blocks = (rows + rows_per_block - 1) / rows_per_block;
    double start_time = cycle_timer::current_seconds();
    dynamic_multiply_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(rows,
        rows_per_thread, row_ptr, col_ind, values, x, y);
    cudaThreadSynchronize();
    double end_time = cycle_timer::current_seconds();
    printf("gpu dynamic multiply kernel: %f\n", end_time - start_time);

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
 * @brief   Caller function for new sparse matrix multiplication in CUDA.
 *
 * @param   m   The matrix to multiply.
 * @param   v   The vector to multiply.
 *
 * @return  The result of matrix vector multiplication of m*v.
 */
template <typename T>
vector<T> cuda_new_multiply(const csr_matrix<T> &m, const vector<T> &v) {
    int rows = m.row_size();
    int cols = m.col_size();
    int nonzeros = m.nonzeros();
    assert(cols == v.size());

    // Malloc device space
    int *row_ptr, *col_ind, *row_ind;
    T *values, *x, *y;
    cudaMalloc(&row_ptr, sizeof(int) * (rows + 1));
    cudaMalloc(&col_ind, sizeof(int) * nonzeros);
    cudaMalloc(&row_ind, sizeof(int) * nonzeros);
    cudaMalloc(&values, sizeof(T) * nonzeros);
    cudaMalloc(&x, sizeof(T) * cols);
    cudaMalloc(&y, sizeof(T) * cols);

    // Transfer data from host to device
    cudaMemcpy(row_ptr, m.row_ptr_data(), sizeof(int) * (rows + 1),
        cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind, m.col_ind_data(), sizeof(int) * nonzeros,
        cudaMemcpyHostToDevice);
    cudaMemcpy(row_ind, m.row_ind_data(), sizeof(int) * nonzeros,
        cudaMemcpyHostToDevice);
    cudaMemcpy(values, m.values_data(), sizeof(T) * nonzeros,
        cudaMemcpyHostToDevice);
    cudaMemcpy(x, v.data(), sizeof(T) * cols, cudaMemcpyHostToDevice);

    // Run kernel
    const int blocks = (nonzeros + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    double start_time = cycle_timer::current_seconds();
    new_multiply_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(rows,
        row_ptr, col_ind, row_ind, values, x, y);
    cudaThreadSynchronize();
    double end_time = cycle_timer::current_seconds();
    printf("gpu new multiply kernel: %f\n", end_time - start_time);


    // Transfer result back from device to host
    vector<T> result(cols);
    cudaMemcpy(result.data(), y, sizeof(T) * cols, cudaMemcpyDeviceToHost);

    // Release device space
    cudaFree(row_ptr);
    cudaFree(col_ind);
    cudaFree(row_ind);
    cudaFree(values);
    cudaFree(x);
    cudaFree(y);

    return result;
}

/**
 * @brief   Caller function for naive Lanczos algorithm in CUDA.
 *
 * @param   m       The matrix to do operations on.
 * @param   v       The initial vector with norm 1.
 * @param   steps   The iteration times for lanczos algorithm.
 *
 * @return  The tridiagonal matrix result of lanczos algorithm.
 */
template <typename T>
symm_tridiag_matrix<T> cuda_lanczos(const csr_matrix<T> &m,
    const vector<T> &v, const int steps) {
    symm_tridiag_matrix<T> result(steps + 1);

    int rows = m.row_size();
    int cols = m.col_size();
    int nonzeros = m.nonzeros();
    const int blocks = (rows + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    assert(rows == cols);
    assert(cols == v.size());

    // Malloc device space
    int *row_ptr, *col_ind;
    T *values, *x, *x_prev, *y, *scratch;
    cudaMalloc(&row_ptr, sizeof(int) * (rows + 1));
    cudaMalloc(&col_ind, sizeof(int) * nonzeros);
    cudaMalloc(&values, sizeof(T) * nonzeros);
    cudaMalloc(&x, sizeof(T) * cols);
    cudaMalloc(&x_prev, sizeof(T) * cols);
    cudaMalloc(&y, sizeof(T) * cols);
    cudaMalloc(&scratch, sizeof(T) * blocks);

    // Transfer data from host to device
    cudaMemcpy(row_ptr, m.row_ptr_data(), sizeof(int) * (rows + 1),
        cudaMemcpyHostToDevice);
    cudaMemcpy(col_ind, m.col_ind_data(), sizeof(int) * nonzeros,
        cudaMemcpyHostToDevice);
    cudaMemcpy(values, m.values_data(), sizeof(T) * nonzeros,
        cudaMemcpyHostToDevice);
    cudaMemcpy(x, v.data(), sizeof(T) * cols, cudaMemcpyHostToDevice);

    // Run kernel
    for (int i = 0; i < steps; i++) {
        // y_i = M*x_i
        naive_multiply_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(rows,
            row_ptr, col_ind, values, x, y);
        cudaThreadSynchronize();
        // alpha_i <- y_i*x_i
        T product = device_dot_product(rows, x, y, scratch);
        result.alpha(i) = product;
        // y_i <- y_i - alpha_i*x_i - beta_i*x_(i-1)
        saxpy_inplace_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(rows,
            y, x, -product);
        cudaThreadSynchronize();
        if (i > 0) {
            saxpy_inplace_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(rows,
                y, x_prev, -result.beta(i - 1));
            cudaThreadSynchronize();
        }
        std::swap(x, x_prev);
        // beta_(i+1) <- ||y_i||
        result.beta(i) = T(sqrt(device_dot_product(rows, y, y, scratch)));
        // x_(i+1) <- y_i / beta_(i+1)
        multiply_inplace_kernel<T><<<blocks, THREADS_PER_BLOCK>>>(rows, y,
            1 / result.beta(i));
        cudaThreadSynchronize();
        std::swap(x, y);
    }

    // Release device space
    cudaFree(row_ptr);
    cudaFree(col_ind);
    cudaFree(values);
    cudaFree(x);
    cudaFree(x_prev);
    cudaFree(y);
    cudaFree(scratch);

    result.resize(steps);
    return result;
}

/**
 * @brief   Lanczos algorithm for eigendecomposition in CUDA.
 * 
 * @param   matrix  CSR matrix to decompose
 * @param   k       number of largest eigenvalues to compute
 * @param   steps   maximum steps for the iteration
 * @tparam  T       matrix element data type
 * @return  list of eigenvalues
 */
template <typename T>
vector<T> cuda_lanczos_eigen(const csr_matrix<T> &matrix, int k, int steps) {
    int cols = matrix.col_size();
    assert(cols > 0);
    vector<T> v(cols, 0);
    v[0] = 1;
    symm_tridiag_matrix<T> tridiag = cuda_lanczos(matrix, v, steps);
    return lanczos_no_spurious(tridiag, k);
}

template void cuda_multiply_inplace(vector<float> &v, const float &k);
template void cuda_add_inplace(vector<float> &v, const float &k);
template void cuda_saxpy_inplace(vector<float> &y, const float &a, const vector<float> &x);
template float cuda_dot_product(const vector<float> &v1, const vector<float> &v2);
template float cuda_l2_norm(const vector<float> &v);

template void cuda_multiply_inplace(vector<double> &v, const double &k);
template void cuda_add_inplace(vector<double> &v, const double &k);
template void cuda_saxpy_inplace(vector<double> &y, const double &a, const vector<double> &x);
template double cuda_dot_product(const vector<double> &v1, const vector<double> &v2);
template double cuda_l2_norm(const vector<double> &v);

template vector<float> cuda_naive_multiply(const csr_matrix<float> &m, const vector<float> &v);
template vector<float> cuda_warp_multiply(const csr_matrix<float> &m, const vector<float> &v);
template vector<float> cuda_cusparse_multiply(const csr_matrix<float> &m, const vector<float> &v);

template vector<double> cuda_naive_multiply(const csr_matrix<double> &m, const vector<double> &v);
template vector<double> cuda_warp_multiply(const csr_matrix<double> &m, const vector<double> &v);
template vector<double> cuda_cusparse_multiply(const csr_matrix<double> &m, const vector<double> &v);

template vector<float> cuda_lanczos_eigen(const csr_matrix<float> &matrix, int k, int steps);
template vector<double> cuda_lanczos_eigen(const csr_matrix<double> &matrix, int k, int steps);
