from cython.parallel import prange
import numpy as np

from libc cimport math as cm
cimport numpy as cnp

from scify cimport _machine as m
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
    if np.isscalar(x):
        return _clausen(x)

    cdef:
        cnp.ndarray[cnp.npy_float64, ndim=1] arr = np.ravel(x)
        int n = arr.size

    if threaded:
        clausen_p(arr, n)
    else:
        clausen_s(arr, n)

    return arr.reshape(np.shape(x))


cdef void clausen_p(double[::1] x, int size) nogil:
    """Parallel"""
    cdef int i
    for i in prange(size, nogil=True):
        x[i] = _clausen(x[i])


cdef void clausen_s(double[::1] x, int size) nogil:
    """Single Thread"""
    cdef int i
    for i in range(size):
        x[i] = _clausen(x[i])


cdef double _clausen(double x) nogil:
    cdef:
        double x_cut = m.M_PI * m.DBL_EPSILON
        double sgn = 1.0

    if x < 0:
        x = -x
        sgn = -1.0

    x = angle_restrict_pos_err(x)

    if x > m.M_PI:
        x = (6.28125 - x) + 1.9353071795864769253e-03
        sgn = -sgn

    if x == 0.0:
        return 0
    elif x < x_cut:
        return x * (1.0 - cm.log(x)) * sgn
    else:
        return x * (cheb_eval(constants, 2 * (x * x / (m.M_PI ** 2) - 0.5)) - cm.log(x)) * sgn
