#! /usr/bin/env python

from __future__ import with_statement

from StringIO import StringIO
import numpy as np
from scipy.sparse import dia_matrix

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='eigendecomposition for symmetric tridiag matrices')
    parser.add_argument('path', help='matrix file')
    args = parser.parse_args()

    with open(args.path) as data:
        n = int(data.readline().strip())
        alpha = np.fromstring(data.readline().strip(), sep=' ')
        beta = np.fromstring(data.readline().strip(), sep=' ')

    tridiag = dia_matrix((n, n))
    tridiag.setdiag(alpha, 0)
    tridiag.setdiag(beta, 1)
    tridiag.setdiag(beta, -1)
    eigen, _ = np.linalg.eig(tridiag.todense())
    eigen.sort()
    output = StringIO()
    np.savetxt(output, eigen[::-1], fmt='%.5f', newline=' ')
    print output.getvalue()
