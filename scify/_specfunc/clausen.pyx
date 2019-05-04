import numpy as np

from libc cimport math as cm
cimport numpy as cnp

from scify cimport _machine as m
from ._results cimport Result, make_r_0, map_dbl_p, map_dbl_s
from .cheb cimport cheb_eval
from .trig cimport angle_restrict_pos_err


cdef:
    double[::1] constants = np.array([
        2.142694363766688447e+00,
        0.723324281221257925e-01,
        0.101642475021151164e-02,
        0.3245250328531645e-04,
        0.133315187571472e-05,
        0.6213240591653e-07,
        0.313004135337e-08,
        0.16635723056e-09,
        0.919659293e-11,
        0.52400462e-12,
        0.3058040e-13,
        0.18197e-14,
        0.1100e-15,
        0.68e-17,
        0.4e-18
    ])


def clausen(x, bint threaded=True):
    x = np.asarray(x, float)
    cdef:
        cnp.ndarray[cnp.npy_float64, ndim=1] arr = x.ravel()
        int n = arr.size

    if n == 1:
        return _clausen(arr[0]).val
    if threaded and n > 1:
        map_dbl_p(_clausen, arr, n)
    else:
        map_dbl_s(_clausen, arr, n)

    return arr.reshape(x.shape)


cdef Result _clausen(double x) nogil:
    cdef:
        Result res = make_r_0()
        Result sr
        double x_cut = m.M_PI * m.DBL_EPSILON
        int sgn = 1

    if x < 0:
        x = -x
        sgn = -1

    sr = angle_restrict_pos_err(x)

    x = sr.val

    if x > m.M_PI:
        x = (6.28125 - x) + 1.9353071795864769253e-03
        sgn = -sgn

    if x == 0.0:
        return res
    elif x < x_cut:
        res.val = x * (1 - cm.log(x))
        res.err = x * m.DBL_EPSILON
    else:
        sr = cheb_eval(constants, 2 * (x * x / (m.M_PI ** 2) - 0.5), -1, 1)
        res.val = x * (sr.val - cm.log(x))
        res.err = x * (sr.err + m.DBL_EPSILON)

    res.val *= sgn
    return res
