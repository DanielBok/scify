import numpy as np

from libc cimport math as cm
cimport numpy as cnp

from scify cimport _machine as m
from ._results cimport (
Result, make_r_0, make_r_nan, map_dbl_p, map_dbl_s,
ComplexResult, make_c_0, mapc_dbl_p, mapc_dbl_s
)
from .clausen cimport _clausen
from .log cimport complex_log

cdef:
    double PI = m.M_PI
    double DBL_EPS = m.DBL_EPSILON


def dilog(x, bint threaded):
    x = np.asarray(x, float)
    cdef:
        cnp.ndarray[cnp.npy_float64, ndim=1] arr = x.ravel()
        int n = arr.size

    if n == 1:
        return _dilog(arr[0]).val
    if threaded and n > 1:
        map_dbl_p(_dilog, arr, n)
    else:
        map_dbl_s(_dilog, arr, n)

    return arr.reshape(x.shape)

cdef Result _dilog(double x) nogil:
    cdef:
        Result res = make_r_0()
        Result d1, d2

    if x >= 0:
        return dilog_xge0(x)
    d1 = dilog_xge0(-x)
    d2 = dilog_xge0(x * x)
    res.val = 0.5 * d2.val - d1.val
    res.err = d1.err + 0.5 * d2.err + 2 * DBL_EPS * cm.fabs(res.val)
    return res

cdef Result dilog_xge0(double x) nogil:
    """Calculates dilog for real :math:`x \geq 0"""
    cdef:
        Result res = make_r_0()
        Result ser
        double log_x = cm.log(x)
        double t1, t2, t3
        int i

    if x > 2:
        ser = dilog_series_2(1. / x)
        t1 = PI * PI / 3
        t2 = ser.val
        t3 = 0.5 * log_x ** 2
        res.val = t1 - t2 - t3
        res.err = DBL_EPS * (cm.fabs(log_x) + cm.fabs(t1) + cm.fabs(t2) + cm.fabs(t3) + 2 * cm.fabs(res.val)) + ser.err

    elif x > 1.01:
        ser = dilog_series_2(1 - 1. / x)
        t1 = PI ** 2 / 6
        t2 = ser.val
        t3 = log_x * (cm.log(1 - 1. / x) + 0.5 * log_x)
        res.val = t1 + t2 - t3
        res.err = DBL_EPS * (cm.fabs(log_x) + cm.fabs(t1) + cm.fabs(t2) + cm.fabs(t3) + 2 * cm.fabs(res.val)) + ser.err

    elif x > 1:
        t2 = cm.log(x - 1)
        t3 = 0

        for i in range(8, 0, -1):
            t1 = (-1) ** (i + 1) * (1 - i * t2) / (i * i)
            t3 = (x - 1) * (t3 + t1)

        res.val = t3 + PI * PI / 6
        res.err = 2 * DBL_EPS * cm.fabs(res.val)

    elif cm.fabs(x - 1) <= m.DBL_EPSILON * 10:
        res.val = PI ** 2 / 6
        res.err = 2 * DBL_EPS * res.val

    elif x > 0.5:
        ser = dilog_series_2(1 - x)
        t1 = PI ** 2 / 6
        t2 = ser.val
        t3 = log_x * cm.log(1 - x)
        res.val = t1 - t2 - t3
        res.err = DBL_EPS * (cm.fabs(log_x) + cm.fabs(t1) + cm.fabs(t2) + cm.fabs(t3) + 2 * cm.fabs(res.val)) + ser.err

    elif x > 0.25:
        return dilog_series_2(x)

    elif x > 0:
        return dilog_series_1(x)

    return res

cdef Result dilog_series_1(double x) nogil:
    cdef:
        Result res = make_r_0()
        double rk2, term = x, total = x
        int k

    for k in range(2, 1000):
        rk2 = ((k - 1.0) / k) ** 2
        term *= x * rk2
        total += term

        if cm.fabs(term / total) < DBL_EPS:
            res.val = total
            res.err = 2 * (cm.fabs(term) + DBL_EPS * cm.fabs(res.val))
            return res

    # Max iteration hit. dilog_series_1 could not converge
    return make_r_nan()

cdef Result dilog_series_2(double x) nogil:
    cdef:
        Result res = make_r_0()
        double total = 0.5 * x, y = x, z = 0
        double ds
        int k

    for k in range(2, 100):
        y *= x
        ds = y / (k * k * (k + 1))
        total += ds
        if k >= 10 and cm.fabs(ds / total) < 0.5 * DBL_EPS:
            break

    res.val = total
    res.err = 2.0 * 100 * DBL_EPS * cm.fabs(total)

    if x > 0.01:
        z = (1 - x) * cm.log(1 - x) / x
    else:
        for k in range(8, 1, -1):
            z = x * (1.0 / k + z)
        z = (x - 1) * (1 + z)

    res.val += z + 1
    res.err += 2.0 * DBL_EPS * cm.fabs(z)
    return res

def dilog_complex(r, theta, bint threaded):
    r = np.asarray(r, float)
    theta = np.asarray(theta, float)

    cdef:
        ComplexResult c
        cnp.ndarray[cnp.npy_float64, ndim=1] r_vec = r.ravel()
        cnp.ndarray[cnp.npy_float64, ndim=1] theta_vec = theta.ravel()
        int n = r_vec.size

    assert r.shape == theta.shape, "Radius of complex vector must have same shape as the angled part"

    if n == 1:
        c = _dilog_complex(r, theta)
        return c.real + 1j * c.imag
    if threaded:
        mapc_dbl_p(_dilog_complex, r_vec, theta_vec, n)
    else:
        mapc_dbl_s(_dilog_complex, r_vec, theta_vec, n)

    return (r_vec + 1j * theta_vec).reshape(r.shape)

cdef ComplexResult _dilog_complex(double r, double theta) nogil:
    cdef:
        ComplexResult c = make_c_0()
        Result real_res
        double x = r * cm.cos(theta)
        double y = r * cm.sin(theta)
        double zeta2 = PI ** 2 / 6
        double r2 = x * x + y * y
        double t1, t2

        double ln_minusz_re, ln_minusz_im, lmz2_re, lmz2_im

    if cm.fabs(y) < 10 * DBL_EPS:
        real_res = _dilog(x)
        c.real, c.real_err = real_res.val, real_res.err
        if x >= 1:
            c.imag = -PI * cm.log(x)
            c.imag_err = 2 * DBL_EPS * cm.fabs(c.imag)

    elif cm.fabs(r2 - 1) <= DBL_EPS:
        t1 = theta * theta / 4
        t2 = PI * cm.fabs(theta) / 2
        c.real = zeta2 + t1 - t2
        c.real_err = 2 * DBL_EPS * (zeta2 + t1 + t2)

        real_res = _clausen(theta)
        c.imag = real_res.val
        c.imag_err = real_res.err

    elif r2 < 1:
        return dilogc_unit_disk(x, y)

    else:
        c = dilogc_unit_disk(x / r2, - y / r2)
        ln_minusz_re = cm.log(r)
        ln_minusz_im = (-1.0 if theta < 0.0 else 1.0) * (cm.fabs(theta) - PI)
        lmz2_re = ln_minusz_re ** 2 - ln_minusz_im ** 2
        lmz2_im = 2.0 * ln_minusz_re * ln_minusz_im

        c.real = -c.real - 0.5 * lmz2_re - zeta2
        c.real_err += 2 * DBL_EPS * (0.5 * cm.fabs(lmz2_re) + zeta2)
        c.imag = -c.imag - 0.5 * lmz2_im
        c.imag_err += 2 * DBL_EPS * cm.fabs(lmz2_im)

    return c

cdef inline ComplexResult dilogc_fundamental(double r, double x, double y) nogil:
    if r > 0.98:
        return dilogc_series_3(r, x, y)
    elif r > 0.25:
        return dilogc_series_2(r, x, y)
    else:
        return dilogc_series_1(r, x, y)

cdef ComplexResult dilogc_unit_disk(double x, double y) nogil:
    cdef:
        ComplexResult c = make_c_0()
        ComplexResult tmp_c
        double r = cm.hypot(x, y)
        double zeta2 = PI ** 2 / 6
        double x_tmp, y_tmp, r_tmp, lnz, lnomz, argz, argomz

    if x > 0.732:  # magic split value
        x_tmp = 1.0 - x
        y_tmp = -y
        r_tmp = cm.hypot(x_tmp, y_tmp)
        tmp_c = dilogc_fundamental(r_tmp, x_tmp, y_tmp)

        lnz = cm.log(r)  # log(|z|)
        lnomz = cm.log(r_tmp)  # log(|1 - z|)
        argz = cm.atan2(y, x)  # arg(z)
        argomz = cm.atan2(y_tmp, x_tmp)  # arg(1 - z)

        c.real = -tmp_c.real + zeta2 - lnz * lnomz + argz * argomz
        c.real_err = tmp_c.real_err + 2 * DBL_EPS * (zeta2 + cm.fabs(lnz * lnomz) + cm.fabs(argz * argomz))
        c.imag = -tmp_c.imag - argz * lnomz - argomz * lnz
        c.imag_err = tmp_c.imag_err + 2 * DBL_EPS * (cm.fabs(argz * lnomz) + cm.fabs(argomz * lnz))

        return c
    else:
        return dilogc_fundamental(r, x, y)

cdef ComplexResult dilogc_series_1(double r, double x, double y) nogil:
    cdef:
        ComplexResult c = make_c_0()
        double cos_theta = x / r
        double sin_theta = y / r
        double alpha = 1 - cos_theta
        double beta = sin_theta
        double ck = cos_theta
        double sk = sin_theta
        double rk = r
        double real = r * ck
        double imag = r * sk
        int k, kmax = 50 + <int> (-22 / cm.log(r))

    for k in range(2, kmax):
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

    c.real = real
    c.real_err = 2 * kmax * DBL_EPS * cm.fabs(real)
    c.imag = imag
    c.imag_err = 2 * kmax * DBL_EPS * cm.fabs(imag)
    return c

cdef ComplexResult dilogc_series_2(double r, double x, double y) nogil:
    cdef:
        ComplexResult c = make_c_0()
        ComplexResult ln_omz, sum_c
        double r2 = r ** 2
        double tx, ty, rx, ry

    if cm.fabs(r) <= m.DBL_EPSILON * 10:
        return c

    sum_c = series_2_c(r, x, y)
    ln_omz = complex_log(1 - x, -y)

    tx = (ln_omz.real * x + ln_omz.imag * y) / r2
    ty = (-ln_omz.real * y + ln_omz.imag * x) / r2
    rx = (1 - x) * tx + y * ty
    ry = (1 - x) * ty - y * tx

    c.real = sum_c.real + rx + 1
    c.imag = sum_c.imag + ry
    c.real_err = sum_c.real_err + 2 * DBL_EPS * (cm.fabs(c.real) + cm.fabs(rx))
    c.imag_err = sum_c.imag_err + 2 * DBL_EPS * (cm.fabs(c.imag) + cm.fabs(ry))

    return c

cdef ComplexResult dilogc_series_3(double r, double x, double y) nogil:
    cdef:
        ComplexResult c = make_c_0()
        double theta = cm.atan2(y, x)
        double cos_theta = x / r
        double sin_theta = y / r
        double omc = 1.0 - cos_theta

        Result claus = _clausen(theta)

        double*re = [
            PI ** 2 / 6 + 0.25 * (theta ** 2 - 2 * PI * cm.fabs(theta)),
            -0.5 * cm.log(2 * omc),
            -0.5,
            -0.5 / omc,
            0,
            0.5 * (2.0 + cos_theta) / (omc ** 2),
            0
        ]
        double*im = [
            claus.val,
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

    c.real = sum_re
    c.real_err = 2 * 6 * DBL_EPS * cm.fabs(sum_re) + cm.fabs(an / nfact)
    c.imag = sum_im
    c.imag_err = 2 * 6 * DBL_EPS * cm.fabs(sum_im) + claus.err + cm.fabs(an / nfact)
    return c

cdef ComplexResult series_2_c(double r, double x, double y) nogil:
    cdef:
        ComplexResult c = make_c_0()
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
        int k, kmax = 30 + <int> (18.0 / (-cm.log(r)))
        double limit = m.DBL_EPSILON ** 2

    for k in range(2, kmax):
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

    c.real = real
    c.real_err = 2 * kmax * DBL_EPS * cm.fabs(real)
    c.imag = imag
    c.imag_err = 2 * kmax * DBL_EPS * cm.fabs(imag)
    return c
