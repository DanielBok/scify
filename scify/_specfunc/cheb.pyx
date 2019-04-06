cimport cython


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef double cheb_eval(double[:] constants, double x) nogil:
    cdef:
        double d = 0.0, dd = 0.0
        int j

    for j in range(len(constants) - 1, 0, -1):
        dd, d = d, 2 * x * d - dd + constants[j]

    return x * d - dd + 0.5 * constants[0]


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef double cheb_eval_(double* constants, double x, int order) nogil:
    cdef:
        double d = 0.0, dd = 0.0
        int j

    for j in range(order - 1, 0, -1):
        dd, d = d, 2 * x * d - dd + constants[j]

    return x * d - dd + 0.5 * constants[0]
