#include <cstdlib>
#include <iostream>
#include <getopt.h>

#include "graph_io.h"
#include "matrix.h"

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

int main(int argc, char *argv[]) {
    parse_option(argc, argv);

    coo_matrix<float> graph = adjacency_matrix_from_graph<float>(node_count, graph_file);
    csr_matrix<float> matrix(graph);

    return 0;
}