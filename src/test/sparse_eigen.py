#! /usr/bin/env python

from StringIO import StringIO
import numpy as np
import scipy.sparse.linalg
from scipy.sparse import coo_matrix

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='eigendecomposition for sparse matrices')
    parser.add_argument('path', help='matrix file')
    parser.add_argument('-n', '--node-count', type=int, metavar='N', help='matrix node count', required=True)
    parser.add_argument('-k', '--eigen-count', type=int, metavar='K', help='number of eigenvalues to compute', required=True)
    args = parser.parse_args()

    edges = np.loadtxt(args.path)
    e, s = edges.shape
    if s < 3:
        row, col = edges[:, 0], edges[:, 1]
        data = np.ones((e,))
    else:
        row, col, data = edges[:, 0], edges[:, 1], edges[:, 2]
    n = args.node_count
    matrix = coo_matrix((data, (row, col)), shape=(n, n))
    eigen, _ = scipy.sparse.linalg.eigsh(matrix, k=args.eigen_count, which='LA')
    eigen.sort()
    output = StringIO()
    np.savetxt(output, eigen[::-1], fmt='%.5f', newline=' ')
    print output.getvalue()
