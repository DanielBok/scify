import numpy as np

from libc cimport math as cm
cimport numpy as cnp

from scify cimport _machine as m
from ._results cimport Result, make_r_0, make_r_nan, map_dbl_p, map_dbl_s
from .cheb cimport cheb_eval

ctypedef double (*DFunc) (double) nogil


cdef double X_CUT = -m.LOG_DBL_MIN

cdef:
    double[::1] db1 = np.array([
        2.4006597190381410194,
        0.1937213042189360089,
        -0.62329124554895770e-02,
        0.3511174770206480e-03,
        -0.228222466701231e-04,
        0.15805467875030e-05,
        -0.1135378197072e-06,
        0.83583361188e-08,
        -0.6264424787e-09,
        0.476033489e-10,
        -0.36574154e-11,
        0.2835431e-12,
        -0.221473e-13,
        0.17409e-14,
        -0.1376e-15,
        0.109e-16,
        -0.9e-18
    ])
    double[::1] db2 = np.array([
        2.5943810232570770282,
        0.2863357204530719834,
        -0.102062656158046713e-01,
        0.6049109775346844e-03,
        -0.405257658950210e-04,
        0.28633826328811e-05,
        -0.2086394303065e-06,
        0.155237875826e-07,
        -0.11731280087e-08,
        0.897358589e-10,
        -0.69317614e-11,
        0.5398057e-12,
        -0.423241e-13,
        0.33378e-14,
        -0.2645e-15,
        0.211e-16,
        -0.17e-17,
        0.1e-18
    ])
    double[::1] db3 = np.array([
        2.707737068327440945,
        0.340068135211091751,
        -0.12945150184440869e-01,
        0.7963755380173816e-03,
        -0.546360009590824e-04,
        0.39243019598805e-05,
        -0.2894032823539e-06,
        0.217317613962e-07,
        -0.16542099950e-08,
        0.1272796189e-09,
        -0.987963460e-11,
        0.7725074e-12,
        -0.607797e-13,
        0.48076e-14,
        -0.3820e-15,
        0.305e-16,
        -0.24e-17
    ])
    double[::1] db4 = np.array([
        2.781869415020523460,
        0.374976783526892863,
        -0.14940907399031583e-01,
        0.945679811437042e-03,
        -0.66132916138933e-04,
        0.4815632982144e-05,
        -0.3588083958759e-06,
        0.271601187416e-07,
        -0.20807099122e-08,
        0.1609383869e-09,
        -0.125470979e-10,
        0.9847265e-12,
        -0.777237e-13,
        0.61648e-14,
        -0.4911e-15,
        0.393e-16,
        -0.32e-17
    ])
    double[::1] db5 = np.array([
        2.8340269546834530149,
        0.3994098857106266445,
        -0.164566764773099646e-1,
        0.10652138340664541e-2,
        -0.756730374875418e-4,
        0.55745985240273e-5,
        -0.4190692330918e-6,
        0.319456143678e-7,
        -0.24613318171e-8,
        0.1912801633e-9,
        -0.149720049e-10,
        0.11790312e-11,
        -0.933329e-13,
        0.74218e-14,
        -0.5925e-15,
        0.475e-16,
        -0.39e-17
    ])
    double[::1] db6 = np.array([
        2.8726727134130122113,
        0.4174375352339027746,
        -0.176453849354067873e-1,
        0.11629852733494556e-2,
        -0.837118027357117e-4,
        0.62283611596189e-5,
        -0.4718644465636e-6,
        0.361950397806e-7,
        -0.28030368010e-8,
        0.2187681983e-9,
        -0.171857387e-10,
        0.13575809e-11,
        -0.1077580e-12,
        0.85893e-14,
        -0.6872e-15,
        0.552e-16,
        -0.44e-17
    ])


def debye_1(x, bint threaded):
    x = np.asarray(x, float)
    cdef:
        cnp.ndarray[cnp.npy_float64, ndim=1] arr = x.ravel()
        int n = arr.size

    if n == 1:
        return _debye_1(arr[0]).val
    elif threaded:
        map_dbl_p(_debye_1, arr, n)
    else:
        map_dbl_s(_debye_1, arr, n)

    return arr.reshape(x.shape)


cdef Result _debye_1(double x) nogil:
    cdef:
        Result res = make_r_0()
        Result c
        double val_infinity = 1.64493406684822644
        int i, nexp
        double total, ex

    if x < 0:
        return make_r_nan()

    elif x < 2 * m.SQRT_DBL_EPSILON:
        res.val = 1 - 0.25 * x + x ** 2 / 36.
        res.err = m.DBL_EPSILON * cm.fabs(res.val)

    elif x <= 4:
        c = cheb_eval(db1, x * x / 8. - 1, -1, 1)
        res.val = c.val - 0.25 * x
        res.err = c.err + 0.25 * x * m.DBL_EPSILON

    elif x <= -(m.M_LN2 + m.LOG_DBL_EPSILON):
        nexp = <int> cm.floor(X_CUT / x)
        ex = cm.exp(-x)
        total = 0.0
        for i in range(nexp, 0, -1):
            total *= ex
            total += (1 + 1 / (x * i)) / i

        res.val = val_infinity / x - total * ex
        res.err = m.DBL_EPSILON * cm.fabs(res.val)

    elif x < X_CUT:
        res.val = (val_infinity - cm.exp(-x) * (x + 1)) / x
        res.err = m.DBL_EPSILON * cm.fabs(res.val)

    else:
        res.val = val_infinity / x
        res.err = m.DBL_EPSILON * cm.fabs(res.val)

    return res


def debye_2(x, bint threaded):
    x = np.asarray(x, float)
    cdef:
        cnp.ndarray[cnp.npy_float64, ndim=1] arr = x.ravel()
        int n = arr.size

    if n == 1:
        return _debye_2(arr[0]).val
    elif threaded:
        map_dbl_p(_debye_2, arr, n)
    else:
        map_dbl_s(_debye_2, arr, n)

    return arr.reshape(x.shape)


cdef Result _debye_2(double x) nogil:
    cdef:
        Result res = make_r_0()
        Result c
        double val_infinity = 4.80822761263837714, x2 = x ** 2
        int i, nexp
        double total, ex, xi

    if x < 0:
        return make_r_nan()

    elif x < 2 * m.M_SQRT2 * m.SQRT_DBL_EPSILON:
        res.val = 1 - x / 3 + x2 / 24
        res.err = m.DBL_EPSILON * res.val

    elif x <= 4:
        c = cheb_eval(db2, x * x / 8. - 1, -1, 1)
        res.val = c.val - x / 3.
        res.err = c.err + m.DBL_EPSILON * x / 3.

    elif x < - (m.M_LN2 - m.LOG_DBL_EPSILON):
        nexp = <int> cm.floor(X_CUT / x)
        ex = cm.exp(-x)
        total = 0.0
        for i in range(nexp, 0, -1):
            total *= ex
            xi = x * i
            total += (1 + 2 / xi + 2 / (xi ** 2)) / i

        res.val = val_infinity / x ** 2 - 2 * total * ex
        res.err = m.DBL_EPSILON * cm.fabs(res.val)

    elif x < X_CUT:
        total = 2 + 2 * x + x2
        res.val = (val_infinity - 2 * total * cm.exp(-x)) / x2
        res.err = m.DBL_EPSILON * cm.fabs(res.val)

    else:
        res.val = val_infinity / x2
        res.err = m.DBL_EPSILON * res.val

    return res


def debye_3(x, bint threaded):
    x = np.asarray(x, float)
    cdef:
        cnp.ndarray[cnp.npy_float64, ndim=1] arr = x.ravel()
        int n = arr.size

    if n == 1:
        return _debye_3(arr[0]).val
    elif threaded:
        map_dbl_p(_debye_3, arr, n)
    else:
        map_dbl_s(_debye_3, arr, n)

    return arr.reshape(x.shape)


cdef Result _debye_3(double x) nogil:
    cdef:
        Result res = make_r_0()
        Result c
        double val_infinity = 19.4818182068004875, x3 = x ** 3
        int i, nexp
        double total, ex, xinv

    if x < 0:
        return make_r_nan()
    elif x < 2.0 * m.M_SQRT2 * m.SQRT_DBL_EPSILON:
        res.val = 1 - 3 * x / 8 + x ** 2 / 20
        res.err = m.DBL_EPSILON * res.val

    elif x <= 4:
        c = cheb_eval(db3, x * x / 8. - 1, -1, 1)
        res.val = c.val - 0.375 * x
        res.err = c.err + m.DBL_EPSILON * 0.375 * x

    elif x < - (m.M_LN2 - m.LOG_DBL_EPSILON):
        nexp = <int>cm.floor(X_CUT / x)
        ex = cm.exp(-x)
        total = 0.0
        for i in range(nexp, 0, -1):
            xinv = 1 / (x * i)
            total *= ex
            total += (((6 * xinv + 6) * xinv + 3) * xinv + 1) / i

        res.val = val_infinity / x ** 3 - 3 * total * ex
        res.err = m.DBL_EPSILON * res.val

    elif x < X_CUT:
        total = 6 + 6 * x + 3 * x ** 2 + x3
        res.val = (val_infinity - 3 * total * cm.exp(-x)) / x3
        res.err = m.DBL_EPSILON * res.val
        
    else:
        res.val = val_infinity / x3
        res.err = m.DBL_EPSILON * res.val
    
    return res


def debye_4(x, bint threaded):
    x = np.asarray(x, float)
    cdef:
        cnp.ndarray[cnp.npy_float64, ndim=1] arr = x.ravel()
        int n = arr.size

    if n == 1:
        return _debye_4(arr[0]).val
    elif threaded:
        map_dbl_p(_debye_4, arr, n)
    else:
        map_dbl_s(_debye_4, arr, n)

    return arr.reshape(x.shape)


cdef Result _debye_4(double x) nogil:
    cdef:
        Result res = make_r_0()
        Result c
        double val_infinity = 99.5450644937635129, x4 = x ** 4
        int i, nexp
        double total, ex, xinv         

    if x < 0:
        return make_r_nan()
    
    elif x < 2.0 * m.M_SQRT2 * m.SQRT_DBL_EPSILON:
        res.val = 1 - 2 * x / 5 + x ** 2 / 18
        res.err = m.DBL_EPSILON * res.val
        
    elif x <= 4:
        c = cheb_eval(db4, x * x / 8. - 1, -1, 1)
        res.val = c.val - 0.4 * x
        res.err = c.err + m.DBL_EPSILON * 0.4 * x

    elif x < - (m.M_LN2 - m.LOG_DBL_EPSILON):
        nexp = <int> cm.floor(X_CUT / x)
        ex = cm.exp(-x)
        total = 0.0
        for i in range(nexp, 0, -1):
            xinv = 1 / (x * i)
            total *= ex
            total += ((((24 * xinv + 24) * xinv + 12) * xinv + 4) * xinv + 1) / i

        res.val = val_infinity / x ** 4 - 4 * total * ex
        res.err = m.DBL_EPSILON * res.val

    elif x < X_CUT:
        total = 24 + 24 * x + 12 * x ** 2 + 4 * x ** 3 + x4
        res.val = (val_infinity - 4 * total * cm.exp(-x)) / x4
        res.err = m.DBL_EPSILON * res.val

    else:
        res.val = val_infinity / x4
        res.err = m.DBL_EPSILON * res.val

    return res

def debye_5(x, bint threaded):
    x = np.asarray(x, float)
    cdef:
        cnp.ndarray[cnp.npy_float64, ndim=1] arr = x.ravel()
        int n = arr.size

    if n == 1:
        return _debye_5(arr[0]).val
    elif threaded:
        map_dbl_p(_debye_5, arr, n)
    else:
        map_dbl_s(_debye_5, arr, n)

    return arr.reshape(x.shape)


cdef Result _debye_5(double x) nogil:
    cdef:
        Result res = make_r_0()
        Result c
        double val_infinity = 610.405837190669483828710757875, x5 = x ** 5
        int i, nexp
        double total, ex, xinv

    if x < 0:
        return make_r_nan()

    elif x < 2.0 * m.M_SQRT2 * m.SQRT_DBL_EPSILON:
        res.val = 1 - 5. * x / 12 + 5. * x ** 2 / 84
        res.err = m.DBL_EPSILON * res.val

    elif x <= 4:
        c = cheb_eval(db5, x * x / 8. - 1, -1, 1)
        res.val = c.val - 5. * x / 12
        res.err = c.err + m.DBL_EPSILON * 5. * x / 12

    elif x < - (m.M_LN2 - m.LOG_DBL_EPSILON):
        nexp = <int> cm.floor(X_CUT / x)
        ex = cm.exp(-x)
        total = 0.0
        for i in range(nexp, 0, -1):
            xinv = 1 / (x * i)
            total *= ex
            total += (((((120 * xinv + 120) * xinv + 60) * xinv + 20) * xinv + 5) * xinv + 1) / i
        res.val = val_infinity / x5 - 5 * total * ex
        res.err = m.DBL_EPSILON * res.val

    elif x < X_CUT:
        total = 120 + 120 * x + 60 * x ** 2 + 20 * x ** 3 + 5 * x ** 4 + x5
        res.val = (val_infinity - 5 * total * cm.exp(-x)) / x5
        res.err = m.DBL_EPSILON * res.val

    else:
        res.val = val_infinity / x5
        res.err = m.DBL_EPSILON * res.val

    return res


def debye_6(x, bint threaded):
    x = np.asarray(x, float)
    cdef:
        cnp.ndarray[cnp.npy_float64, ndim=1] arr = x.ravel()
        int n = arr.size

    if n == 1:
        return _debye_6(arr[0]).val
    elif threaded:
        map_dbl_p(_debye_6, arr, n)
    else:
        map_dbl_s(_debye_6, arr, n)

    return arr.reshape(x.shape)


cdef Result _debye_6(double x) nogil:
    cdef:
        Result res = make_r_0()
        Result c
        double val_infinity = 4356.06887828990661194792541535, x6 = x ** 6
        int i, nexp
        double total, ex, xinv

    if x < 0:
        return make_r_nan()

    elif x < 2.0 * m.M_SQRT2 * m.SQRT_DBL_EPSILON:
        res.val = 1 - 3. * x / 7 + x ** 2 / 16.
        res.err = m.DBL_EPSILON * res.val

    elif x <= 4:
        c = cheb_eval(db6, x * x / 8. - 1, -1, 1)
        res.val = c.val - 3. * x / 7
        res.err = c.err + m.DBL_EPSILON - 3. * x / 7

    elif x < - (m.M_LN2 - m.LOG_DBL_EPSILON):
        nexp = <int> cm.floor(X_CUT / x)
        ex = cm.exp(-x)
        total = 0.0
        for i in range(nexp, 0, -1):
            xinv = 1 / (x * i)
            total *= ex
            total += ((((((720 * xinv + 720) * xinv + 360) * xinv + 120) * xinv + 30) * xinv + 6) * xinv + 1) / i

        res.val =  val_infinity / x6 - 6 * total * ex
        res.err = m.DBL_EPSILON * res.val

    elif x < X_CUT:
        total = 720 + 720 * x + 360 * x ** 2 + 120 * x ** 3 + 30 * x ** 4 + 6 * x ** 5 + x6
        res.val = (val_infinity - 6 * total * cm.exp(-x)) / x6
        res.err = m.DBL_EPSILON * res.val

    else:
        res.val = val_infinity / x6
        res.err = m.DBL_EPSILON * res.val

    return res
