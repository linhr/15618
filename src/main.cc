#include <cstdlib>
#include <iostream>
#include <getopt.h>

#include "graph_io.h"
#include "matrix.h"
#include "linear_algebra.h"
#include "cuda_algebra.h"

using std::cout;
using std::cerr;
using std::endl;

static string graph_file;
static int node_count = 0;

static void usage(const char *program) {
    cout << "usage: " << program << " [options]" << endl;
    cout << "options:" << endl;
    cout << "  -g --graph <file>" << endl;
    cout << "  -n --nodes <n>" << endl;
}

static void parse_option(int argc, char *argv[]) {
    int opt;
    static struct option long_options[] = {
        { "help", 0, 0, 'h' },
        { "graph", 1, 0, 'g' },
        { "nodes", 1, 0, 'n' },
        { 0, 0, 0, 0 },
    };
    while ((opt = getopt_long(argc, argv, "g:n:h?", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'g':
            graph_file = optarg;
            break;
        case 'n':
            node_count = atoi(optarg);
            break;
        case 'h':
        case '?':
        default:
            usage(argv[0]);
            exit(opt == 'h' ? 0 : 1);
        }
    }
    if (graph_file.empty()) {
        cerr << argv[0] << ": missing graph file" << endl;
        exit(1);
    }
    if (node_count <= 0) {
        cerr << argv[0] << ": invalid node count" << endl;
        exit(1);
    }
}

void test() {
    // Due to different order of adding floats, Cuda results may slightly differ from cpu results
    int n = 8192;

    vector<float> x(n, 0);
    vector<float> y(n, 0);
    vector<float> z(n, 0);
    float k = 3.0;

    srandom(1);
    for (int i = 0; i < n; i++) {
        //x[i] = random()/((double) RAND_MAX);
        //y[i] = random()/((double) RAND_MAX);
        x[i] = random() * 10 / RAND_MAX;
        y[i] = random() * 10 / RAND_MAX;
    }
    float cpu_result = dot_product<float>(x, y);
    float gpu_result = cuda_dot_product<float>(x, y);
    printf("Dot_product - cpu: %f, gpu: %f\n", cpu_result, gpu_result);

    for (int i = 0; i < n; i++) {
        x[i] = random() * 10 / RAND_MAX;
        y[i] = x[i];
    }
    multiply_inplace(x, k);
    cuda_multiply_inplace(y, k);
    for (int i = 0; i < n; i++) {
        if (x[i] != y[i]) {
            printf("Multiply inplace disagree\n");
            return;
        }
    }
    printf("Multiply inplace checked\n");

    for (int i = 0; i < n; i++) {
        x[i] = random() * 10 / RAND_MAX;
        y[i] = x[i];
    }
    add_inplace(x, k);
    cuda_add_inplace(y, k);
    for (int i = 0; i < n; i++) {
        if (x[i] != y[i]) {
            printf("Add inplace disagree\n");
            return;
        }
    }
    printf("Add inplace checked\n");

    for (int i = 0; i < n; i++) {
        x[i] = random() * 10 / RAND_MAX;
        y[i] = x[i];
        z[i] = random() * 10 / RAND_MAX;
    }
    saxpy_inplace(x, k, z);
    cuda_saxpy_inplace(y, k, z);
    for (int i = 0; i < n; i++) {
        if (x[i] != y[i]) {
            printf("Saxpy inplace disagree\n");
            return;
        }
    }
    printf("Saxpy inplace checked\n");

    float w(1);
    coo_matrix<float> graph(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int p = random() % 100;
            if (p < 2) {
                graph.add_entry(i, j, w);
            }
        }
    }
    csr_matrix<float> matrix(graph);
    for (int i = 0; i < n; i++) {
        z[i] = random() * 10 / RAND_MAX;
    }
    vector<float> a = multiply(matrix, z);
    vector<float> b = cuda_naive_multiply(matrix, z);
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("Naive mv multiply disagree\n");
            return;
        }
    }
    printf("Naive mv multiply checked\n");
    vector<float> c = cuda_warp_multiply(matrix, z);
    for (int i = 0; i < n; i++) {
        if (a[i] != c[i]) {
            printf("Warp mv multiply disagree\n");
            return;
        }
    }
    printf("Warp mv multiply checked\n");
}

int main(int argc, char *argv[]) {
    parse_option(argc, argv);

    coo_matrix<float> graph = adjacency_matrix_from_graph<float>(node_count, graph_file);
    csr_matrix<float> matrix(graph);

    test();

    return 0;
}
