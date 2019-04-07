from libc cimport math as cm

from scify cimport _machine as m
from ._results cimport make_r


cdef double cheb_eval(double[::1] constants, double x) nogil:
    cdef:
        double d = 0, dd = 0
        int j, n = len(constants)

    for j in range(n - 1, 0, -1):
        dd, d = d, 2 * x * d - dd + constants[j]

    return x * d - dd + 0.5 * constants[0]


cdef double cheb_eval_(double*constants, double x, int length) nogil:
    cdef:
        double d = 0, dd = 0
        int j

    for j in range(length - 1, 0, -1):
        dd, d = d, 2 * x * d - dd + constants[j]

    return x * d - dd + 0.5 * constants[0]


cdef double cheb_eval_mode(double[::1] constants, double x, int a, int b) nogil:
    cdef:
        double d = 0, dd = 0
        double y = (2 * x - a - b) / (b - a)  # a: upper bound, b: lower bound
        double y2 = 2 * y
        int j, n = len(constants)

    for j in range(n - 1, 0, -1):
        dd, d = d, y2 * d - dd + constants[j]

    return y * d - dd + 0.5 * constants[0]


cdef Result cheb_eval_mode_e(double[::1] constants, double x, int a, int b) nogil:
    cdef:
        double d = 0, dd = 0
        double y = (2 * x - a - b) / (b - a)  # a: upper bound, b: lower bound
        size_t length = len(constants)
        double y2 = 2 * y
        double val, err
        int j

    for j in range(length - 1, 0, -1):
        dd, d = d, y2 * d - dd + constants[j]

    val = y * d - dd + 0.5 * constants[0]
    err = m.DBL_EPSILON * cm.fabs(val) + length

    return make_r(val, err)
