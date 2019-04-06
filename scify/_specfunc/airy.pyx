cimport cython
from libc cimport math as cm

from cython.parallel import prange
import numpy as np

from scify cimport _machine as m
from .cheb cimport cheb_eval_mode
from .trig cimport cos_err


cdef:
    double*aif_cs = [
        -0.03797135849666999750,
        0.05919188853726363857,
        0.00098629280577279975,
        0.00000684884381907656,
        0.00000002594202596219,
        0.00000000006176612774,
        0.00000000000010092454,
        0.00000000000000012014,
        0.00000000000000000010
    ]
    double*aig_cs = [
        0.01815236558116127,
        0.02157256316601076,
        0.00025678356987483,
        0.00000142652141197,
        0.00000000457211492,
        0.00000000000952517,
        0.00000000000001392,
        0.00000000000000001
    ]

@cython.cdivision(True)
cdef (double, double) airy_mod_phase(double x) nogil:
    """airy function for x < -1"""
    cdef:
        double z, mod = 0, phase = 0
        double sqx = cm.sqrt(-x)
        double*a1 = [
            0.0065809191761485,
            0.0023675984685722,
            0.0001324741670371,
            0.0000157600904043,
            0.0000027529702663,
            0.0000006102679017,
            0.0000001595088468,
            0.0000000471033947,
            0.0000000152933871,
            0.0000000053590722,
            0.0000000020000910,
            0.0000000007872292,
            0.0000000003243103,
            0.0000000001390106,
            0.0000000000617011,
            0.0000000000282491,
            0.0000000000132979,
            0.0000000000064188,
            0.0000000000031697,
            0.0000000000015981,
            0.0000000000008213,
            0.0000000000004296,
            0.0000000000002284,
            0.0000000000001232,
            0.0000000000000675,
            0.0000000000000374,
            0.0000000000000210,
            0.0000000000000119,
            0.0000000000000068,
            0.0000000000000039,
            0.0000000000000023,
            0.0000000000000013,
            0.0000000000000008,
            0.0000000000000005,
            0.0000000000000003,
            0.0000000000000001,
            0.0000000000000001
        ]
        double*a2 = [
            -0.07125837815669365,
            -0.00590471979831451,
            -0.00012114544069499,
            -0.00000988608542270,
            -0.00000138084097352,
            -0.00000026142640172,
            -0.00000006050432589,
            -0.00000001618436223,
            -0.00000000483464911,
            -0.00000000157655272,
            -0.00000000055231518,
            -0.00000000020545441,
            -0.00000000008043412,
            -0.00000000003291252,
            -0.00000000001399875,
            -0.00000000000616151,
            -0.00000000000279614,
            -0.00000000000130428,
            -0.00000000000062373,
            -0.00000000000030512,
            -0.00000000000015239,
            -0.00000000000007758,
            -0.00000000000004020,
            -0.00000000000002117,
            -0.00000000000001132,
            -0.00000000000000614,
            -0.00000000000000337,
            -0.00000000000000188,
            -0.00000000000000105,
            -0.00000000000000060,
            -0.00000000000000034,
            -0.00000000000000020,
            -0.00000000000000011,
            -0.00000000000000007,
            -0.00000000000000004,
            -0.00000000000000002
        ]
        double*b1 = [
            -0.01562844480625341,
            0.00778336445239681,
            0.00086705777047718,
            0.00015696627315611,
            0.00003563962571432,
            0.00000924598335425,
            0.00000262110161850,
            0.00000079188221651,
            0.00000025104152792,
            0.00000008265223206,
            0.00000002805711662,
            0.00000000976821090,
            0.00000000347407923,
            0.00000000125828132,
            0.00000000046298826,
            0.00000000017272825,
            0.00000000006523192,
            0.00000000002490471,
            0.00000000000960156,
            0.00000000000373448,
            0.00000000000146417,
            0.00000000000057826,
            0.00000000000022991,
            0.00000000000009197,
            0.00000000000003700,
            0.00000000000001496,
            0.00000000000000608,
            0.00000000000000248,
            0.00000000000000101,
            0.00000000000000041,
            0.00000000000000017,
            0.00000000000000007,
            0.00000000000000002
        ]
        double*b2 = [
            0.00440527345871877,
            -0.03042919452318455,
            -0.00138565328377179,
            -0.00018044439089549,
            -0.00003380847108327,
            -0.00000767818353522,
            -0.00000196783944371,
            -0.00000054837271158,
            -0.00000016254615505,
            -0.00000005053049981,
            -0.00000001631580701,
            -0.00000000543420411,
            -0.00000000185739855,
            -0.00000000064895120,
            -0.00000000023105948,
            -0.00000000008363282,
            -0.00000000003071196,
            -0.00000000001142367,
            -0.00000000000429811,
            -0.00000000000163389,
            -0.00000000000062693,
            -0.00000000000024260,
            -0.00000000000009461,
            -0.00000000000003716,
            -0.00000000000001469,
            -0.00000000000000584,
            -0.00000000000000233,
            -0.00000000000000093,
            -0.00000000000000037,
            -0.00000000000000015,
            -0.00000000000000006,
            -0.00000000000000002
        ]

    if x < -2:
        z = 16.0 / (x ** 3) + 1
        mod = cheb_eval_mode(a1, z, 37, -1, 1)
        phase = cheb_eval_mode(a2, z, 36, -1, 1)
    elif x <= -1:
        z = (16 / (x ** 3) + 9) / 7
        mod = cheb_eval_mode(b1, z, 33, -1, 1)
        phase = cheb_eval_mode(b2, z, 32, -1, 1)

    mod += 0.3125
    phase -= 0.625

    return cm.sqrt(mod / sqx), m.M_PI_4 - x * sqx * phase


@cython.boundscheck(False)
@cython.wraparound(False)
def airy_Ai(x):
    r"""
    Computes the Airy function of the first kind. This is defined as

    .. math::

        Ai(x) = (1/\pi) \int_0^\infty \cos(\t^3/3 + xt) dt

    For more information, checkout the article on `Wikipedia <https://en.wikipedia.org/wiki/Airy_function>`_

    Parameters
    ----------
    x: {array_like, scalar}
        Numerical vector

    Returns
    -------
    arraylike or scalar
        Values as defined by the Airy function
    """
    cdef:
        double[:] arr
        int i
        size_t n

    if np.isscalar(x):
        return _airy_Ai(x)

    arr = np.ravel(x)
    n = len(arr)
    for i in prange(n, nogil=True):
        arr[i] = _airy_Ai(arr[i])

    return np.reshape(arr, np.shape(x))

@cython.cdivision(True)
@cython.nonecheck(False)
cdef double _airy_Ai(double x) nogil:
    cdef:
        double mod, theta, z

    if x < -1:
        mod, theta = airy_mod_phase(x)
        return mod * cos_err(theta)
    elif x <= 1:
        z = x ** 3
        mod = cheb_eval_mode(aif_cs, z, 9, -1, 1)
        theta = cheb_eval_mode(aig_cs, z, 8, -1, 1)
        return 0.375 + (mod - x * (0.25 + theta))
    else:
        z = airy_aie(x) * cm.exp(-2 * x * cm.sqrt(x) / 3)
        if cm.fabs(z) < m.DBL_MIN:
            with gil:
                raise ValueError("Underflow encountered")
        return z

@cython.cdivision(True)
@cython.nonecheck(False)
cdef double airy_aie(double x) nogil:
    """airy function for x >= 1"""
    cdef:
        double*constant = [
            -0.0187519297793867540198,
            -0.0091443848250055004725,
            0.0009010457337825074652,
            -0.0001394184127221491507,
            0.0000273815815785209370,
            -0.0000062750421119959424,
            0.0000016064844184831521,
            -0.0000004476392158510354,
            0.0000001334635874651668,
            -0.0000000420735334263215,
            0.0000000139021990246364,
            -0.0000000047831848068048,
            0.0000000017047897907465,
            -0.0000000006268389576018,
            0.0000000002369824276612,
            -0.0000000000918641139267,
            0.0000000000364278543037,
            -0.0000000000147475551725,
            0.0000000000060851006556,
            -0.0000000000025552772234,
            0.0000000000010906187250,
            -0.0000000000004725870319,
            0.0000000000002076969064,
            -0.0000000000000924976214,
            0.0000000000000417096723,
            -0.0000000000000190299093,
            0.0000000000000087790676,
            -0.0000000000000040927557,
            0.0000000000000019271068,
            -0.0000000000000009160199,
            0.0000000000000004393567,
            -0.0000000000000002125503,
            0.0000000000000001036735,
            -0.0000000000000000509642,
            0.0000000000000000252377,
            -0.0000000000000000125793
        ]
        double sqx = cm.sqrt(x)
        double z = 2 / (x * sqx) - 1

    return (0.28125 + cheb_eval_mode(constant, z, 36, -1, 1)) / cm.sqrt(sqx)

@cython.boundscheck(False)
@cython.wraparound(False)
def airy_Ai_scaled(x):
    r"""
    Computes a scaled version of the Airy function of the first kind.

    This is defined as

    .. math::

        Ai_s = \left.
        \begin{cases}
            (1/\pi) \int_0^\infty \cos(\t^3/3 + xt) dt, & x < 0 \\
            \exp^{1.5 x^1.5} (1/\pi) \int_0^\infty \cos(\t^3/3 + xt) dt, & x \geq 0
        \end{cases}
        \right}

    For more information, checkout the article on `Wikipedia <https://en.wikipedia.org/wiki/Airy_function>`_

    Parameters
    ----------
    x: {array_like, scalar}
        Numerical vector

    Returns
    -------
    arraylike or scalar
        Values as defined by the Airy function
    """
    cdef:
        double[:] arr
        int i
        size_t n

    if np.isscalar(x):
        return _airy_Ai_scaled(x)

    arr = np.ravel(x)
    n = len(arr)
    for i in prange(n, nogil=True):
        arr[i] = _airy_Ai_scaled(arr[i])

    return np.reshape(arr, np.shape(x))


@cython.cdivision(True)
@cython.nonecheck(False)
cdef double _airy_Ai_scaled(double x) nogil:
    cdef:
        double mod, theta, z, val, scale

    if x < -1:
        mod, theta = airy_mod_phase(x)
        return mod * cos_err(theta)
    elif x < 1:
        z = x ** 3
        mod = cheb_eval_mode(aif_cs, z, 9, -1, 1)
        theta = cheb_eval_mode(aig_cs, z, 8, -1, 1)
        val = 0.375 + (mod - x * (0.25 + theta))

        if x > 0:
            return val * cm.exp(2. / 3 * cm.sqrt(z))
        return val
    else:
        return airy_aie(x)
