from libc cimport math as cm

from scify cimport _machine as m
from ._results cimport Result, make_r


cdef Result cheb_eval(double[::1] constants, double x, int a, int b) nogil:
    cdef:
        double d = 0, dd = 0, err = 0
        double y = (2. * x - a - b) / (b - a)
        double y2 = 2 * y
        double temp
        size_t i, n = len(constants)

    for i in range(n - 1, 0, -1):
        temp = d
        d = y2 * d - dd + constants[i]
        err += cm.fabs(y2*temp) + cm.fabs(dd) + cm.fabs(constants[i])
        dd = temp

    temp = d
    d = y * d - dd + 0.5 * constants[0]
    err += cm.fabs(y * temp) + cm.fabs(dd) + 0.5 * cm.fabs(constants[0])

    return make_r(d, m.DBL_EPSILON * err + cm.fabs(constants[n - 1]))


cdef Result cheb_eval_mode(double[::1] constants, double x, int a, int b) nogil:
    cdef:
        double d = 0, dd = 0
        double y = (2 * x - a - b) / (b - a)
        double y2 = 2 * y
        double val, err
        size_t i, length = len(constants)

    for i in range(length - 1, 0, -1):
        dd, d = d, y2 * d - dd + constants[i]

    val = y * d - dd + 0.5 * constants[0]
    err = m.DBL_EPSILON * cm.fabs(val) + length

    return make_r(val, err)
