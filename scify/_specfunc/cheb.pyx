cimport cython

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef double cheb_eval(double*constants, double x, int length) nogil:
    cdef:
        double d = 0, dd = 0
        int j

    for j in range(length - 1, 0, -1):
        dd, d = d, 2 * x * d - dd + constants[j]

    return x * d - dd + 0.5 * constants[0]

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef double cheb_eval_mode(double*constants, double x, int length, int a, int b) nogil:
    cdef:
        double d = 0, dd = 0
        double y = (2 * x - a - b) / (b - a)  # a: upper bound, b: lower bound
        double y2 = 2 * y
        int j

    for j in range(length - 1, 0, -1):
        dd, d = d, y2 * d - dd + constants[j]

    return y * d - dd + 0.5 * constants[0]
