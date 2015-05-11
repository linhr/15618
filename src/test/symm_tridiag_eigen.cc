#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "../eigen.h"
#include "../graph_io.h"
#include "../utils.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cout << "usage: " << argv[0] << " <file>" << endl;
        cout << "  <file> symmetric tridiagonal matrix file" << endl;
        exit(1);
    }
    string filename(argv[1]);

    vector<float> eigen = tqlrat_eigen(symm_tridiag_matrix_from_file<float>(filename));
    sort(eigen.rbegin(), eigen.rend());
    print_vector(eigen);

    return 0;
}
