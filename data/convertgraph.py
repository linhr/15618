#!/usr/bin/env python

import networkx as nx

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='convert network edge list file')
    parser.add_argument('-i', '--input', metavar='FILE', required=True, help='input file')
    parser.add_argument('-o', '--output', metavar='FILE', required=True, help='output file')
    parser.add_argument('-d', '--delimiter', metavar='S', default='\t', help='delimiter')
    args = parser.parse_args()

    graph = nx.read_edgelist(args.input, comments='#', delimiter=args.delimiter, nodetype=int)
    graph = nx.convert_node_labels_to_integers(graph)
    nx.write_edgelist(nx.DiGraph(graph), args.output, data=False)
