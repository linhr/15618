#include <cmath>
#include <cstdlib>
#include <vector>

#include "../cuda_algebra.h"
#include "../linear_algebra.h"
#include "../utils.h"

using std::cout;
using std::endl;
using std::vector;

template <typename T>
void test_vector_add(int n, T k = 1.0) {
    cout << "*** testing vector-scalar inplace addition ***" << endl;
    vector<T> x = random_vector<T>(n);
    vector<T> y = x;

    add_inplace(x, k);
    cuda_add_inplace(y, k);
    T diff = diff_vector(x, y);

    cout << "CPU-GPU difference: " << diff << endl << endl;
}

template <typename T>
void test_scalar_multiply(int n, T k = 1.0) {
    cout << "*** testing vector-scalar inplace multiplication ***" << endl;
    vector<T> x = random_vector<T>(n);
    vector<T> y = x;

    multiply_inplace(x, k);
    cuda_multiply_inplace(y, k);
    T diff = diff_vector(x, y);

    cout << "CPU-GPU difference: " << diff << endl << endl;
}

template <typename T>
void test_saxpy(int n, T k = 1.0) {
    cout << "*** testing inplace SAXPY ***" << endl;
    vector<T> x = random_vector<T>(n);
    vector<T> y = x;
    vector<T> z = random_vector<T>(n);

    saxpy_inplace(x, k, z);
    cuda_saxpy_inplace(y, k, z);
    T diff = diff_vector(x, y);

    cout << "CPU-GPU difference: " << diff << endl << endl;
}

template <typename T>
void test_dot_product(int n) {
    cout << "*** testing dot product ***" << endl;
    vector<T> x = random_vector<T>(n);
    vector<T> y = x;

    T cpu_result = dot_product(x, y);
    T gpu_result = cuda_dot_product(x, y);
    T diff = std::abs(cpu_result - gpu_result);

    cout << "CPU-GPU difference: " << diff << endl << endl;
}

template <typename T>
void test_l2_norm(int n) {
    cout << "*** testing l2-norm ***" << endl;
    vector<T> x = random_vector<T>(n);

    T cpu_result = l2_norm(x);
    T gpu_result = cuda_l2_norm(x);
    T diff = std::abs(cpu_result - gpu_result);

    cout << "CPU-GPU difference: " << diff << endl << endl;
}

template <typename T>
static void test(int n) {
    test_vector_add<T>(n, 3.0);
    test_scalar_multiply<T>(n, 3.0);
    test_saxpy<T>(n, 3.0);
    test_dot_product<T>(n);
    test_l2_norm<T>(n);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "usage: " << argv[0] << " <n>" << endl;
        cout << "  <n> vector length" << endl;
        exit(1);
    }
    int n = atoi(argv[1]);
    test<float>(n);
    return 0;
}
