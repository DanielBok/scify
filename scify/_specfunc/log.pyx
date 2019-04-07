from libc cimport math as cm


cdef (double, double) complex_log(double zr, double zi) nogil:
    cdef:
        double ax, ay, min, max

    if zr == 0 and zi == 0:
        return cm.NAN, cm.NAN

    ax = cm.fabs(zr)
    ay = cm.fabs(zi)
    min = cm.fmin(ax, ay)
    max = cm.fmax(ax, ay)

    return cm.log(max) + 0.5 * cm.log(1 + (min / max) ** 2), cm.atan2(zi, zr)
