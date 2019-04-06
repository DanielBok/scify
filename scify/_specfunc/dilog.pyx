import numpy as np
from cython.parallel import prange

import cython
from libc cimport math as cm

from scify cimport _machine as m
from .clausen cimport _clausen
from .log cimport complex_log

cdef:
    double PI = m.M_PI


@cython.boundscheck(False)
@cython.wraparound(False)
def dilog(x):
    r"""
    Computes the dilogarithm for a real argument. In Lewin’s notation this is  :math:`Li_2(x)`,
    the real part of the dilogarithm of a real :math:`x`. It is defined by the integral
    representation :math:`Li_2(x) = -\Re \int_0^x \frac{\log(1-s)}{s} ds`.

    Note that :math:`\Im(Li_2(x)) = 0 \forall x \leq 1` and :math:`\Im(Li_2(x)) = -\pi\log(x) \forall x > 1`.

    Parameters
    ----------
    x: {array_like, scalar}
        Numeric vector input

    Returns
    -------
    {array_like, scalar}
        Real Dilog output
    """
    cdef:
        double[:] arr
        int i
        size_t n

    if np.isscalar(x):
        return _dilog(x)

    arr = np.ravel(x)
    n = len(arr)
    for i in prange(n, nogil=True):
        arr[i] = _dilog(arr[i])

    return np.reshape(arr, np.shape(x))


cdef double _dilog(double x) nogil:
    cdef:
        double d1, d2

    if x > 0:
        return dilog_xge0(x)
    else:
        d1 = dilog_xge0(-x)
        d2 = dilog_xge0(x * x)

        return -d1 + 0.5 * d2


@cython.cdivision(True)
cdef double dilog_xge0(double x) nogil:
    """Calculates dilog for real :math:`x \geq 0"""
    cdef:
        double log_x = cm.log(x)
        double t1, t2, t3
        double eps, lne
        int i

    if x > 2:
        t1 = PI * PI / 3
        t2 = dilog_series_2(1 / x)
        t3 = 0.5 * log_x ** 2
        return t1 - t2 -t3

    elif x > 1.01:
        t1 = PI ** 2 / 6
        t2 = dilog_series_2(1 - 1 / x)
        t3 = log_x * (cm.log(1 - 1/x) + 0.5 * log_x)
        return t1 + t2 - t3

    elif x > 1:
        t2 = cm.log(x - 1)
        t3 = 0

        for i in range(8, 0, -1):
            t1 = (-1) ** (i + 1) * (1 - i * t2) / (i * i)
            t3 = (x - 1) * (t3 + t1)

        return t3 + PI * PI / 6

    elif cm.fabs(x - 1) <= m.DBL_EPSILON * 10:
        return PI ** 2 / 6

    elif x > 0.5:
        t1 = PI ** 2 / 6
        t2 = dilog_series_2(1 - x)
        t3 = log_x * cm.log(1 - x)

        return t1 - t2 - t3

    elif x > 0.25:
        return dilog_series_2(x)

    elif x > 0:
        return dilog_series_1(x)

    else:
        return 0


@cython.cdivision(True)
cdef double dilog_series_1(double x) nogil:
    cdef:
        double rk2, term = x, total = x
        int k

    for k in range(2, 1000):
        rk2 = ((k - 1.0) / k) ** 2
        term *= x * rk2
        total += term

        if cm.fabs(term / total) < m.DBL_EPSILON:
            return total

    with gil:
        raise StopIteration('Max iteration hit. dilog_series_1 could not converge')



@cython.cdivision(True)
cdef double dilog_series_2(double x) nogil:
    cdef:
        double ds, total = 0.5 * x, y = x, z = 0.0
        int k

    for k in range(2, 100):
        y *= x
        ds = y / (k * k * (k + 1))
        total += ds
        if k >= 10 and cm.fabs(ds / total) < 0.5 * m.DBL_EPSILON:
            break

    if x > 0.01:
        z = (1 - x) * cm.log(1 - x) / x
    else:
        for k in range(8, 1, -1):
            z = x * (1.0 / k + z)
        z = (x - 1) * (1 + z)

    return total + z + 1


@cython.boundscheck(False)
@cython.wraparound(False)
def dilog_complex(r, theta):
    r"""
    This function computes the full complex-valued dilogarithm for the complex argument
    :math=:`z = r \exp(i \theta)`.

    Parameters
    ----------
    r: {array_like, scalar}
        The modulus of the complex vector or scalar
    theta: {array_like, scalar}
        The argument of the complex vector or scalar

    Returns
    -------
    {array_like, scalar}
        Complex Dilog output
    """
    cdef:
        double[:] r_vec, theta_vec
        size_t n
        int i

    if np.isscalar(r) and np.isscalar(theta):
        return complex(*_dilog_complex(r, theta))

    assert np.shape(r) == np.shape(theta)

    r_vec = np.ravel(r)
    theta_vec = np.ravel(theta)
    n = len(r_vec)

    for i in prange(n, nogil=True):
        r_vec[i], theta_vec[i] = _dilog_complex(r_vec[i], theta_vec[i])

    return np.reshape([complex(r_vec[i], theta_vec[i]) for i in range(n)], np.shape(r))


@cython.cdivision(True)
@cython.nonecheck(False)
cdef (double, double) _dilog_complex(double r, double theta) nogil:
    cdef:
        double x = r * cm.cos(theta)
        double y = r * cm.sin(theta)
        double zeta2 = PI ** 2 / 6
        double r2 = x * x + y * y
        double real, imag

        # intermediaries
        double ln_minusz_re, ln_minusz_im, lmz2_re, lmz2_im

    if cm.fabs(y) < 10 * m.DBL_EPSILON:
        imag =  -PI * cm.log(x) if x >= 1 else 0.0
        return _dilog(x), imag

    elif cm.fabs(r2 - 1) <= m.DBL_EPSILON:
        real = zeta2 + (theta * theta / 4) - (PI * cm.fabs(theta) / 2)
        return real, _clausen(theta)

    elif r2 < 1:
        return dilogc_unit_disk(x, y)

    else:
        real, imag = dilogc_unit_disk(x / r2, - y / r2)
        ln_minusz_re = cm.log(r)
        ln_minusz_im = (-1.0 if theta < 0.0 else 1.0) * (cm.fabs(theta) - PI)
        lmz2_re = ln_minusz_re ** 2 - ln_minusz_im ** 2
        lmz2_im = 2.0 * ln_minusz_re * ln_minusz_im

        real = -real - 0.5 * lmz2_re - zeta2
        imag = -imag - 0.5 * lmz2_im
        return real, imag


cdef (double, double) dilogc_fundamental(double r, double x, double y) nogil:
    if r > 0.98:
        return dilogc_series_3(r, x, y)
    elif r > 0.25:
        return dilogc_series_2(r, x, y)
    else:
        return dilogc_series_1(r, x, y)


@cython.cdivision(True)
@cython.nonecheck(False)
cdef (double, double) dilogc_unit_disk(double x, double y) nogil:
    cdef:
        double r = cm.hypot(x, y)
        double x_tmp, y_tmp, r_tmp, real, imag, a, b, c, d

    if x > 0.732:  # magic split value
        x_tmp = 1.0 - x
        y_tmp = -y
        r_tmp = cm.hypot(x_tmp, y_tmp)

        real, imag = dilogc_fundamental(r_tmp, x_tmp, y_tmp)
        a = cm.log(r)
        b = cm.log(r_tmp)
        c = cm.atan2(y, x)
        d = cm.atan2(y_tmp, x_tmp)

        real = -real + PI ** 2 / 6 - a * b + c * d
        imag = -imag - b * c - a * d
        return real, imag
    else:
        return dilogc_fundamental(r, x, y)


@cython.cdivision(True)
@cython.nonecheck(False)
cdef (double, double) dilogc_series_1(double r, double x, double y) nogil:
    cdef:
        double cos_theta = x / r
        double sin_theta = y / r
        double alpha = 1 - cos_theta
        double beta = sin_theta
        double ck = cos_theta
        double sk = sin_theta
        double rk = r
        double real = r * ck
        double imag = r * sk
        int k

    for k in range(2, 50 + <int>(-22 / cm.log(r))):
        ck_tmp = ck
        ck = ck - (alpha * ck + beta * sk)
        sk = sk - (alpha * sk - beta * ck_tmp)
        rk *= r
        dr = rk / (k * k) * ck
        di = rk / (k * k) * sk
        real += dr
        imag += di
        if cm.fabs((dr * dr + di * di) / (real ** 2 + imag ** 2)) < m.DBL_EPSILON ** 2:
            break

    return real, imag


@cython.cdivision(True)
@cython.nonecheck(False)
cdef (double, double) dilogc_series_2(double r, double x, double y) nogil:
    cdef:
        double real, imag
        double om_r, om_i
        double tx, ty, rx, ry

    if cm.fabs(r) <= m.DBL_EPSILON * 10:
        return .0, .0
    real, imag = series_2_c(r, x, y)

    om_r, om_i = complex_log(1 - x, -y)
    tx = (om_r * x + om_i * y) / (r ** 2)
    ty = (-om_r * y + om_i * x) / (r ** 2)
    rx = (1.0 - x) * tx + y * ty
    ry = (1.0 - x) * ty - y * tx

    return real + rx + 1, imag + ry


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef (double, double) dilogc_series_3(double r, double x, double y) nogil:
    cdef:
        double theta = cm.atan2(y, x)
        double cos_theta = x / r
        double sin_theta = y / r
        double omc = 1.0 - cos_theta
        double* re = [
            PI ** 2 / 6 + 0.25 * (theta ** 2 - 2 * PI * cm.fabs(theta)),
            -0.5 * cm.log(2 * omc),
            -0.5,
            -0.5 / omc,
            0,
            0.5 * (2.0 + cos_theta) / (omc ** 2),
            0
        ]
        double* im = [
            _clausen(theta),
            -cm.atan2(-sin_theta, omc),
            0.5 * sin_theta / omc,
            0,
            -0.5 * sin_theta / (omc ** 2),
            0,
            0.5 * sin_theta / (omc ** 5) * (8.0 * omc - sin_theta * sin_theta * (3.0 + cos_theta))
        ]
        double sum_re = re[0], sum_im = im[0]
        double a = cm.log(r), an = 1.0, nfact = 1.0
        int n

    for n in range(1, 7):
        an *= a
        nfact *= n
        sum_re += an / nfact * re[n]
        sum_im += an / nfact * im[n]

    return sum_re, sum_im


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef (double, double) series_2_c(double r, double x, double y) nogil:
    cdef:
        double cos_theta = x / r
        double sin_theta = y / r
        double alpha = 1 - cos_theta
        double beta = sin_theta
        double ck = cos_theta
        double sk = sin_theta
        double rk = r
        double real = 0.5 * r * ck
        double imag = 0.5 * r * sk
        double ck_tmp, di, dr
        int k
        double limit = m.DBL_EPSILON ** 2

    for k in range(2, 30 + <int>(18.0 / (-cm.log(r)))):
        ck_tmp = ck
        ck = ck - (alpha * ck + beta * sk)
        sk = sk - (alpha * sk - beta * ck_tmp)
        rk *= r
        dr = rk / (k * k * (k + 1.0)) * ck
        di = rk / (k * k * (k + 1.0)) * sk
        real += dr
        imag += di
        if cm.fabs((dr ** 2 + di ** 2) / (real ** 2 + imag ** 2)) < limit:
            break

    return real, imag
