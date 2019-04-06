cimport cython
from cython.parallel import prange
from libc cimport math as cm

import numpy as np

from scify cimport _machine as m
from .cheb cimport cheb_eval
from .trig cimport angle_restrict_pos_err


@cython.boundscheck(False)
@cython.cdivision(True)
def clausen(x):
    r"""
    The Clausen function is defined by the following integral,

    .. math::

        Cl_2(x) = - \int_0^x \log(2 \sin(t/2)) dt

    See the `Wikipedia <https://en.wikipedia.org/wiki/Clausen_function>`_ article
    for more information.

    Parameters
    ----------
    x: {array_like, scalar}
        Numeric vector input

    Returns
    -------
    {array_like, scalar}
        Clausen output
    """
    cdef:
        double[::1] arr
        int i
        size_t n

    if np.isscalar(x):
        return _clausen(x)

    arr = np.ravel(x)
    n = len(arr)
    for i in prange(n, nogil=True):
        arr[i] = _clausen(arr[i])

    return np.reshape(arr, np.shape(x))


cdef double _clausen(double x) nogil:
    cdef:
        double* constants = [
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
        ]
        double x_cut = m.M_PI * m.DBL_EPSILON
        double sgn = 1.0
        double t

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
        return x * (cheb_eval(constants, 2 * (x * x / (m.M_PI ** 2) - 0.5), 15) - cm.log(x)) * sgn
