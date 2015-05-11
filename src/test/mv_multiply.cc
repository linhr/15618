#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "../cuda_algebra.h"
#include "../graph_io.h"
#include "../linear_algebra.h"
#include "../matrix.h"
#include "../utils.h"
#include "../cycle_timer.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

template <typename T>
static void test(const string &filename, int n) {
    coo_matrix<T> graph = adjacency_matrix_from_graph<T>(n, filename);
    csr_matrix<T> matrix(graph);
    vector<T> x = random_vector<T>(n);
    double start_time, end_time;

    start_time = cycle_timer::current_seconds();
    vector<T> a = multiply(matrix, x);
    end_time = cycle_timer::current_seconds();
    cout << "CPU multiply: " << end_time - start_time << " sec" << endl;
    cout << endl;

    start_time = cycle_timer::current_seconds();
    vector<T> b = cuda_naive_multiply(matrix, x);
    end_time = cycle_timer::current_seconds();
    cout << "GPU naive multiply: " << end_time - start_time << " sec" << endl;
    cout << "CPU-GPU difference: " << diff_vector(a, b) << endl << endl;

    start_time = cycle_timer::current_seconds();
    vector<T> c = cuda_warp_multiply(matrix, x);
    end_time = cycle_timer::current_seconds();
    cout << "GPU warp multiply: " << end_time - start_time << " sec" << endl;
    cout << "CPU-GPU difference: " << diff_vector(a, c) << endl << endl;

    start_time = cycle_timer::current_seconds();
    vector<T> d = cuda_cusparse_multiply(matrix, x);
    end_time = cycle_timer::current_seconds();
    cout << "GPU cusparse multiply: " << end_time - start_time << " sec" << endl;
    cout << "CPU-GPU difference: " << diff_vector(a, d) << endl << endl;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cout << "usage: " << argv[0] << " <file> <n> [<t>]" << endl;
        cout << "  <file> matrix coordinate list file" << endl;
        cout << "  <n> matrix node count" << endl;
        cout << "  <t> data type ('float' or 'double')" << endl;
        exit(1);
    }
    string filename(argv[1]);
    int n = atoi(argv[2]);
    string type("float");
    if (argc > 3) {
        type = argv[3];
    }

    if (type == "float") {
        test<float>(filename, n);
    }
    else if (type == "double") {
        test<double>(filename, n);
    }
    else {
        cout << "invalid data type" << endl;
        exit(1);
    }
    return 0;
}
