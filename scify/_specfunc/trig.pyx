cimport cython
from libc cimport math as cm
from scify cimport _machine as m


@cython.cdivision(True)
@cython.nonecheck(False)
cdef double angle_restrict_pos_err(double theta) nogil:
    cdef:
        double two_pi = 2 * m.M_PI,
        double y = 2 * cm.floor(theta / two_pi)
        double r = theta - y * (2 * two_pi)

    if r > two_pi:
        r -= two_pi
    elif r < 0:
        r += two_pi

    if cm.fabs(theta) > 0.0625 / m.DBL_EPSILON:
        return cm.NAN

    return r
