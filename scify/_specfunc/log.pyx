import numpy as np

from libc cimport math as cm
cimport numpy as cnp

from scify cimport _machine as m
from ._results cimport (
ComplexResult, make_c_0, make_c_nan, mapc_dbl_p, mapc_dbl_s
)

def complex_log(zr, zi, bint threaded):
    zr = np.asarray(zr, float)
    zi = np.asarray(zi, float)

    cdef:
        ComplexResult c
        cnp.ndarray[cnp.npy_float64, ndim=1] r_vec = zr.ravel()
        cnp.ndarray[cnp.npy_float64, ndim=1] i_vec = zi.ravel()
        int n = r_vec.size

    assert zr.shape == zi.shape, "Real part of complex vector must have same shape as the imaginary part"

    if n == 1:
        c = _complex_log(r_vec[0], i_vec[0])
        return c.real + 1j * c.imag
    if threaded:
        mapc_dbl_p(_complex_log, r_vec, i_vec, n)
    else:
        mapc_dbl_s(_complex_log, r_vec, i_vec, n)

    return (r_vec + 1j * i_vec).reshape(zr.shape)

cdef ComplexResult _complex_log(double zr, double zi) nogil:
    cdef:
        ComplexResult res = make_c_0()
        double ax, ay, min_, max_

    if zr == 0 and zi == 0:
        return make_c_nan()

    ax = cm.fabs(zr)
    ay = cm.fabs(zi)
    min_ = cm.fmin(ax, ay)
    max_ = cm.fmax(ax, ay)

    res.real = cm.log(max_) + 0.5 * cm.log(1 + (min_ / max_) ** 2)
    res.real_err = 2 * m.DBL_EPSILON * cm.fabs(res.real)
    res.imag = cm.atan2(zi, zr)
    res.real_err = m.DBL_EPSILON * cm.fabs(res.real)

    return res
