from libc cimport math as cm

from scify cimport _machine as m
from ._results cimport ComplexResult, make_c_0, make_c_nan


cdef ComplexResult complex_log(double zr, double zi) nogil:
    cdef:
        ComplexResult res = make_c_0()
        double ax, ay, min, max

    if zr == 0 and zi == 0:
        return make_c_nan()

    ax = cm.fabs(zr)
    ay = cm.fabs(zi)
    min = cm.fmin(ax, ay)
    max = cm.fmax(ax, ay)

    res.real = cm.log(max) + 0.5 * cm.log(1 + (min / max) ** 2)
    res.real_err = 2 * m.DBL_EPSILON * cm.fabs(res.real)
    res.imag = cm.atan2(zi, zr)
    res.real_err = m.DBL_EPSILON * cm.fabs(res.real)

    return res
