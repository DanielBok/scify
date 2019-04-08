import numpy as np

from libc cimport math as cm
cimport numpy as cnp

from scify cimport _machine as m
from ._results cimport (
Result, make_r_0, make_r_nan,
ComplexResult, make_c_0, make_c_nan, mapc_dbl_p, mapc_dbl_s
)
from .cheb cimport cheb_eval

cdef:
    double DBL_EPS = m.DBL_EPSILON
    double[::1] lopx = np.array([
        2.16647910664395270521272590407,
        -0.28565398551049742084877469679,
        0.01517767255690553732382488171,
        -0.00200215904941415466274422081,
        0.00019211375164056698287947962,
        -0.00002553258886105542567601400,
        2.9004512660400621301999384544e-06,
        -3.8873813517057343800270917900e-07,
        4.7743678729400456026672697926e-08,
        -6.4501969776090319441714445454e-09,
        8.2751976628812389601561347296e-10,
        -1.1260499376492049411710290413e-10,
        1.4844576692270934446023686322e-11,
        -2.0328515972462118942821556033e-12,
        2.7291231220549214896095654769e-13,
        -3.7581977830387938294437434651e-14,
        5.1107345870861673561462339876e-15,
        -7.0722150011433276578323272272e-16,
        9.7089758328248469219003866867e-17,
        -1.3492637457521938883731579510e-17,
        1.8657327910677296608121390705e-18
    ])
    double[::1] lopxmx = np.array([
        -1.12100231323744103373737274541,
        0.19553462773379386241549597019,
        -0.01467470453808083971825344956,
        0.00166678250474365477643629067,
        -0.00018543356147700369785746902,
        0.00002280154021771635036301071,
        -2.8031253116633521699214134172e-06,
        3.5936568872522162983669541401e-07,
        -4.6241857041062060284381167925e-08,
        6.0822637459403991012451054971e-09,
        -8.0339824424815790302621320732e-10,
        1.0751718277499375044851551587e-10,
        -1.4445310914224613448759230882e-11,
        1.9573912180610336168921438426e-12,
        -2.6614436796793061741564104510e-13,
        3.6402634315269586532158344584e-14,
        -4.9937495922755006545809120531e-15,
        6.8802890218846809524646902703e-16,
        -9.5034129794804273611403251480e-17,
        1.3170135013050997157326965813e-17
    ])

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
    res.real_err = 2 * DBL_EPS * cm.fabs(res.real)
    res.imag = cm.atan2(zi, zr)
    res.real_err = DBL_EPS * cm.fabs(res.real)

    return res

cdef Result log_1plusx(double x) nogil:
    cdef:
        Result res = make_r_0()
        Result c
        double y = 0
        int i

    if x <= -1:
        return make_r_nan()
    elif cm.fabs(x) < m.ROOT6_DBL_EPSILON:
        for i in range(10, 1, -1):
            y = x * (y + (-1.) ** (i + 1) / i)
        res.val = x * (1 + y)
        res.err = DBL_EPS * cm.fabs(res.val)
    elif cm.fabs(x) < 0.5:
        c = cheb_eval(lopx, 0.5 * (8 * x + 1)/ (x + 2.), -1, 1)
        res.val = x * c.val
        res.err = cm.fabs(x * c.err)
    else:
        res.val = cm.log(1 + x)
        res.err = DBL_EPS * cm.fabs(res.val)
    return res


cdef Result log_1plusx_mx(double x) nogil:
    cdef:
        Result res = make_r_0()
        Result c
        double y = 0
        int i

    if x <= -1:
        return make_r_nan()
    elif cm.fabs(x) < m.ROOT6_DBL_EPSILON:
        for i in range(10, 1, -1):
            y = x * (y + (-1.) ** (i + 1) / i)
        res.val = x * y
        res.err = DBL_EPS * cm.fabs(res.val)
    elif cm.fabs(x) < 0.5:
        c = cheb_eval(lopxmx, 0.5 * (8 * x + 1)/ (x + 2.), -1, 1)
        res.val = x * x * c.val
        res.err = x * x * c.err
    else:
        y = cm.log(1 + x)
        res.val = y - x
        res.err = DBL_EPS * (cm.fabs(y) + cm.fabs(x))
    return res
