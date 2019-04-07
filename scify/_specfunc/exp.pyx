import warnings

cimport cython
from libc cimport math as cm

from scify cimport _machine as m
from ._results cimport Result, make_r

@cython.cdivision(True)
@cython.nonecheck(False)
cdef Result exp_mult_err(double x, double dx, double y, double dy) nogil:
    cdef:
        double ay = cm.fabs(y), ex = cm.exp(x)
        double ly, lnr, a, b, err, val

    if y == 0:
        return make_r(0, cm.fabs(dy * ex))
    elif 0.5 * m.LOG_DBL_MIN < x < 0.5 * m.LOG_DBL_MAX and \
            1.2 * m.SQRT_DBL_MIN < ay < 0.8 * m.SQRT_DBL_MAX:
        val = y * ex
        err = ex * (cm.fabs(dy) + cm.fabs(y * dx)) + 2 * m.DBL_EPSILON * cm.fabs(val)
        return make_r(val, err)
    else:
        ly = cm.log(ay)
        lnr = x + ly

        if lnr > m.LOG_DBL_MAX - 0.01:
            with gil:
                warnings.warn('Overflow encountered in exp_mult_err')
            return make_r(cm.NAN, cm.NAN)
        elif lnr < m.LOG_DBL_MIN + 0.01:
            with gil:
                warnings.warn('Underflow encountered in exp_mult_err')
            return make_r(cm.NAN, cm.NAN)

        a = cm.exp(x - cm.floor(x))
        b = cm.exp(ly - cm.floor(ly))

        val = m.sign(y) * a * b
        err = a * b * (2 * m.DBL_EPSILON + cm.fabs(dy / y) + cm.fabs(dx))
        return make_r(val, err)
