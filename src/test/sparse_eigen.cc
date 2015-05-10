#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../eigen.h"
#include "../graph_io.h"
#include "../matrix.h"
#include "../utils.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        cout << "usage: " << argv[0] << " <file> <n> <k>" << endl;
        cout << "  <file> matrix coordinate list file" << endl;
        cout << "  <n> matrix node count" << endl;
        cout << "  <k> number of eigenvalues to compute" << endl;
        exit(1);
    }
    string filename(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    coo_matrix<float> graph = adjacency_matrix_from_graph<float>(n, filename);
    csr_matrix<float> matrix(graph);
    vector<float> eigen = lanczos_eigen(matrix, k, 2 * k + 1);
    sort(eigen.rbegin(), eigen.rend());
    print_vector(eigen);

    return 0;
}
