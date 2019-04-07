from cython.parallel import prange
import numpy as np

from libc cimport math as cm

from scify cimport _machine as m
from .cheb cimport cheb_eval_mode
from .trig cimport cos_err, sin_err


cdef:
    double[::1] a1 = np.array([
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
    ])
    double[::1] a2 = np.array([
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
    ])
    double[::1] b1 = np.array([
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
    ])
    double[::1] b2 = np.array([
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
    ])
    double[::1] aie = np.array([
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
    ])

    double[::1] aif = np.array([
        -0.03797135849666999750,
        0.05919188853726363857,
        0.00098629280577279975,
        0.00000684884381907656,
        0.00000002594202596219,
        0.00000000006176612774,
        0.00000000000010092454,
        0.00000000000000012014,
        0.00000000000000000010
    ])
    double[::1] aig = np.array([
        0.01815236558116127,
        0.02157256316601076,
        0.00025678356987483,
        0.00000142652141197,
        0.00000000457211492,
        0.00000000000952517,
        0.00000000000001392,
        0.00000000000000001
    ])
    double[::1] bif = np.array([
        -0.01673021647198664948,
        0.10252335834249445610,
        0.00170830925073815165,
        0.00001186254546774468,
        0.00000004493290701779,
        0.00000000010698207143,
        0.00000000000017480643,
        0.00000000000000020810,
        0.00000000000000000018
    ])
    double[::1] big = np.array([
        0.02246622324857452,
        0.03736477545301955,
        0.00044476218957212,
        0.00000247080756363,
        0.00000000791913533,
        0.00000000001649807,
        0.00000000000002411,
        0.00000000000000002
    ])
    double[::1] bif2 = np.array([
        0.0998457269381604100,
        0.4786249778630055380,
        0.0251552119604330118,
        0.0005820693885232645,
        0.0000074997659644377,
        0.0000000613460287034,
        0.0000000003462753885,
        0.0000000000014288910,
        0.0000000000000044962,
        0.0000000000000000111
    ])
    double[::1] big2 = np.array([
        0.033305662145514340,
        0.161309215123197068,
        0.0063190073096134286,
        0.0001187904568162517,
        0.0000013045345886200,
        0.0000000093741259955,
        0.0000000000474580188,
        0.0000000000001783107,
        0.0000000000000005167,
        0.0000000000000000011
    ])
    double[::1] bip = np.array([
        -0.08322047477943447,
        0.01146118927371174,
        0.00042896440718911,
        -0.00014906639379950,
        -0.00001307659726787,
        0.00000632759839610,
        -0.00000042226696982,
        -0.00000019147186298,
        0.00000006453106284,
        -0.00000000784485467,
        -0.00000000096077216,
        0.00000000070004713,
        -0.00000000017731789,
        0.00000000002272089,
        0.00000000000165404,
        -0.00000000000185171,
        0.00000000000059576,
        -0.00000000000012194,
        0.00000000000001334,
        0.00000000000000172,
        -0.00000000000000145,
        0.00000000000000049,
        -0.00000000000000011,
        0.00000000000000001
    ])
    double[::1] bip2 = np.array([
        -0.113596737585988679,
        0.0041381473947881595,
        0.0001353470622119332,
        0.0000104273166530153,
        0.0000013474954767849,
        0.0000001696537405438,
        -0.0000000100965008656,
        -0.0000000167291194937,
        -0.0000000045815364485,
        0.0000000003736681366,
        0.0000000005766930320,
        0.0000000000621812650,
        -0.0000000000632941202,
        -0.0000000000149150479,
        0.0000000000078896213,
        0.0000000000024960513,
        -0.0000000000012130075,
        -0.0000000000003740493,
        0.0000000000002237727,
        0.0000000000000474902,
        -0.0000000000000452616,
        -0.0000000000000030172,
        0.0000000000000091058,
        -0.0000000000000009814,
        -0.0000000000000016429,
        0.0000000000000005533,
        0.0000000000000002175,
        -0.0000000000000001737,
        -0.0000000000000000010
    ])


cdef (double, double) airy_mod_phase(double x) nogil:
    """airy function for x < -1"""
    cdef:
        double z, mod = 0, phase = 0
        double sqx = cm.sqrt(-x)

    if x < -2:
        z = 16. / (x ** 3) + 1
        mod = cheb_eval_mode(a1, z, -1, 1)
        phase = cheb_eval_mode(a2, z, -1, 1)
    elif x <= -1:
        z = (16. / (x ** 3) + 9) / 7
        mod = cheb_eval_mode(b1, z, -1, 1)
        phase = cheb_eval_mode(b2, z, -1, 1)

    mod += 0.3125
    phase -= 0.625

    return cm.sqrt(mod / sqx), m.M_PI_4 - x * sqx * phase


cdef double airy_aie(double x) nogil:
    """airy function of the first kind for x >= 1"""
    cdef:
        double sqx = cm.sqrt(x)
        double z = 2 / (x * sqx) - 1

    return (0.28125 + cheb_eval_mode(aie, z, -1, 1)) / cm.sqrt(sqx)


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
    array_like or scalar
        Values from the Airy function
    """
    cdef:
        double[::1] arr
        long i, n

    if np.isscalar(x):
        return _airy_Ai(x)

    arr = np.ravel(x)
    n = len(arr)
    for i in prange(n, nogil=True):
        arr[i] = _airy_Ai(arr[i])

    return np.reshape(arr, np.shape(x))


cdef double _airy_Ai(double x) nogil:
    cdef:
        double mod, theta, z

    if x < -1:
        mod, theta = airy_mod_phase(x)
        return mod * cos_err(theta)
    elif x <= 1:
        z = x ** 3
        mod = cheb_eval_mode(aif, z, -1, 1)
        theta = cheb_eval_mode(aig, z, -1, 1)
        return 0.375 + (mod - x * (0.25 + theta))
    else:
        z = airy_aie(x) * cm.exp(-2 * x * cm.sqrt(x) / 3)
        if cm.fabs(z) < m.DBL_MIN:
            with gil:
                raise ValueError("Underflow encountered")
        return z


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
    array_like or scalar
        Values from the Airy function
    """
    cdef:
        double[::1] arr
        long i, n

    if np.isscalar(x):
        return _airy_Ai_scaled(x)

    arr = np.ravel(x)
    n = len(arr)
    for i in prange(n, nogil=True):
        arr[i] = _airy_Ai_scaled(arr[i])

    return np.reshape(arr, np.shape(x))


cdef double _airy_Ai_scaled(double x) nogil:
    cdef:
        double mod, theta, z, val, scale

    if x < -1:
        mod, theta = airy_mod_phase(x)
        return mod * cos_err(theta)
    elif x < 1:
        z = x ** 3
        mod = cheb_eval_mode(aif, z, -1, 1)
        theta = cheb_eval_mode(aig, z, -1, 1)
        val = 0.375 + (mod - x * (0.25 + theta))

        if x > 0:
            return val * cm.exp(2. / 3 * cm.sqrt(z))
        return val
    else:
        return airy_aie(x)


cdef double airy_bie(double x) nogil:
    """airy function of the second kind for x >= 2"""
    cdef:
        double sqx = cm.sqrt(x)
        double y = cm.sqrt(sqx)
        double z

    if x < 4:
        z = 8.7506905708484345 / (x * sqx) -2.0938363213560543
        return (0.625 + cheb_eval_mode(bip, z, -1, 1)) / y
    else:
        z = 16 / (x * sqx) - 1
        return (0.625 + cheb_eval_mode(bip2, z, -1, 1)) / y


def airy_Bi(x):
    r"""
    Computes the Airy function of the second kind. This is defined as

    .. math::

        Ai(x) = (1/\pi) \int_0^\infty \cos(\t^3/3 + xt) dt

    For more information, checkout the article on `Wikipedia <https://en.wikipedia.org/wiki/Airy_function>`_

    Parameters
    ----------
    x: {array_like, scalar}
        Numerical vector

    Returns
    -------
    array_like or scalar
        Values from the Airy function
    """
    cdef:
        double[::1] arr
        long i, n

    if np.isscalar(x):
        return _airy_Bi(x)

    arr = np.ravel(x)
    n = len(arr)
    for i in prange(n, nogil=True):
        arr[i] = _airy_Bi(arr[i])

    return np.reshape(arr, np.shape(x))


cdef double _airy_Bi(double x) nogil:
    cdef:
        double mod, theta, z, y, s

    if x < -1:
        mod, theta = airy_mod_phase(x)
        return mod * sin_err(theta)
    elif x < 1:
        z = x ** 3
        mod = cheb_eval_mode(bif, z, -1, 1)
        theta = cheb_eval_mode(big, z, -1, 1)
        return 0.625 + mod + x * (0.4375 + theta)
    elif x <= 2:
        z = (2. * x ** 3 - 9) / 7
        mod = cheb_eval_mode(bif2, z, -1, 1)
        theta = cheb_eval_mode(big2, z, -1, 1)
        return 1.125 + mod + x * (0.625 + theta)
    else:
        y = 2 * x * cm.sqrt(x) / 3

        if y > m.LOG_DBL_MAX - 1:
            with gil:
                raise ValueError("overflow in airy_Bi")

        return airy_bie(x) * cm.exp(y)


def airy_Bi_scaled(x):
    r"""
    Computes a scaled version of the Airy function of the second kind.

    This is defined as

    .. math::

        Bi_s = \left.
        \begin{cases}
            (1/\pi) \int_0^\infty \left[ e^{-(t^3/3) + xt} + \sin((t^3/3) + xt) \right] dt, & x < 0 \\
            \exp^{1.5 x^1.5} (1/\pi) \int_0^\infty \left[ e^{-(t^3/3) + xt} + \sin((t^3/3) + xt) \right] dt, & x \geq 0
        \end{cases}
        \right}

    For more information, checkout the article on `Wikipedia <https://en.wikipedia.org/wiki/Airy_function>`_

    Parameters
    ----------
    x: {array_like, scalar}
        Numerical vector

    Returns
    -------
    array_like or scalar
        Values from the Airy function
    """
    cdef:
        double[::1] arr
        long i, n

    if np.isscalar(x):
        return _airy_Bi_scaled(x)

    arr = np.ravel(x)
    n = len(arr)
    for i in prange(n, nogil=True):
        arr[i] = _airy_Bi_scaled(arr[i])

    return np.reshape(arr, np.shape(x))


cdef double _airy_Bi_scaled(double x) nogil:
    cdef:
        double mod, theta, z, s

    if x < -1:
        mod, theta = airy_mod_phase(x)
        return mod * sin_err(theta)
    elif x < 1:
        z = x ** 3
        s = 1 if x <= 0 else cm.exp(-2./3 * cm.sqrt(z))
        mod = cheb_eval_mode(bif, z, -1, 1)
        theta = cheb_eval_mode(big, z, -1, 1)
        return s * (0.625 + mod + x * (0.4375 + theta))
    elif x <= 2:
        z = (2. * x ** 3 - 9) / 7
        s = cm.exp(-2./3 * x ** 1.5)
        mod = cheb_eval_mode(bif2, z, -1, 1)
        theta = cheb_eval_mode(big2, z, -1, 1)
        return s * (1.125 + mod + x * (0.625 + theta))
    else:
        return airy_bie(x)
