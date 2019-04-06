cimport cython


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef double cheb_eval(double* constants, double x, int order) nogil:
    cdef:
        double d = 0.0, dd = 0.0
        int j

    for j in range(order - 1, 0, -1):
        dd, d = d, 2 * x * d - dd + constants[j]

    return x * d - dd + 0.5 * constants[0]


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef double cheb_eval_mode(double* constants, double x, int order, int a, int b) nogil:
    cdef:
        double d = 0.0, dd = 0.0
        double y  = (2.*x - a - b) / (b - a)
        double y2 = 2.0 * y
        int j

    for j in range(order, 0, -1):
        dd, d = d, y2 * d - dd + constants[j]

    return y * d + 0.5 * constants[0]
