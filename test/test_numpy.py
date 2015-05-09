import numpy
import sys
import timeit

try:
    import numpy.core._dotblas
    print 'FAST BLAS'
except ImportError:
    print 'slow blas'

print "version:", numpy.__version__
print "maxint:", sys.maxint
print

x = numpy.random.random((2000,2000))

setup = "import numpy; x = numpy.random.random((2000,2000))"
count = 3

t = timeit.Timer("numpy.linalg.eig(x)", setup=setup)
print "dot:", t.timeit(count)/count, "sec"
