import numpy as np

from libc cimport math as cm

from scify cimport _machine as m
from ._results cimport Result, make_r
from .cheb cimport cheb_eval

cdef:
    double PI = m.M_PI
    double[::1] cos_constants = np.array([
        0.165391825637921473505668118136,
        -0.00084852883845000173671196530195,
        -0.000210086507222940730213625768083,
        1.16582269619760204299639757584e-6,
        1.43319375856259870334412701165e-7,
        -7.4770883429007141617951330184e-10,
        -6.0969994944584252706997438007e-11,
        2.90748249201909353949854872638e-13,
        1.77126739876261435667156490461e-14,
        -7.6896421502815579078577263149e-17,
        -3.7363121133079412079201377318e-18
    ])
    double[::1] sin_constants = np.array([
        -0.3295190160663511504173,
        0.0025374284671667991990,
        0.0006261928782647355874,
        -4.6495547521854042157541e-06,
        -5.6917531549379706526677e-07,
        3.7283335140973803627866e-09,
        3.0267376484747473727186e-10,
        -1.7400875016436622322022e-12,
        -1.0554678305790849834462e-13,
        5.3701981409132410797062e-16,
        2.5984137983099020336115e-17,
        -1.1821555255364833468288e-19
    ])


cdef Result angle_restrict_pos_err(double theta) nogil:
    cdef:
        Result res = make_r(0, 0)
        double two_pi = 2 * PI
        double y = 2 * cm.floor(theta / two_pi)
        double r = theta - y * 2 * two_pi
        double delta

    if r > two_pi:
        r -= two_pi
    elif r < 0:
        r += two_pi

    res.val = r

    if cm.fabs(theta) > 0.0625 / m.DBL_EPSILON:
        return make_r(cm.NAN, cm.NAN)
    elif cm.fabs(theta) > 0.0625 / m.SQRT_DBL_EPSILON:
        res.err = m.DBL_EPSILON * cm.fabs(res.val - theta)
    else:
        delta = cm.fabs(res.val - theta)
        res.err = 2 * m.DBL_EPSILON * cm.fmin(delta, PI)

    return res


cdef Result cos_err(const double x, const double dx) nogil:
    cdef:
        Result res = make_r(0, 0)
        double abs_x = cm.fabs(x)
        double x2, t, y, z
        int sgn = 1
        int octant

    if abs_x < m.ROOT4_DBL_EPSILON:
        x2 = x * x
        res.val = 1 - 0.5 * x2
        res.err = cm.fabs(x2 * x2 / 12)
    else:
        y = cm.floor(abs_x / (0.25 * PI))
        octant = <int>(y - cm.ldexp(cm.floor(cm.ldexp(y, -3)), 3))

        if octant % 2 == 1:
            octant += 1
            octant &= 7
            y += 1

        if octant > 3:
            octant -= 4
            sgn *= -1

        if octant > 1:
            sgn *= -1

        z = abs_x - y * 7.85398125648498535156e-1 - y * 3.77489470793079817668e-8 - y * 2.69515142907905952645e-15

        t = 8 * cm.fabs(z) / PI - 1
        if octant == 0:
            res.val = 1 - 0.5 * z * z * (1 - z * z * cheb_eval(cos_constants, t, -1, 1).val)
        else:
            res.val = z * (1 + z * z * cheb_eval(sin_constants, t, -1, 1).val)

        res.val *= sgn

        z = cm.fabs(res.val)
        if abs_x > 1/m.DBL_EPSILON:
            res.err = z
        elif abs_x > 100/m.SQRT_DBL_EPSILON:
            res.err = 2.0 * abs_x * m.DBL_EPSILON * z
        elif abs_x > 0.1/m.SQRT_DBL_EPSILON:
            res.err = 2.0 * m.SQRT_DBL_EPSILON * z
        else:
            res.err = 2.0 * m.DBL_EPSILON * z

    res.err += cm.fabs(cm.sin(x) * dx) + m.DBL_EPSILON * res.val

    return res


cdef Result sin_err(const double x, const double dx) nogil:
    cdef:
        Result res = make_r(0, 0)
        double abs_x = cm.fabs(x)
        double x2, t, y, z
        int sgn = m.sign(x)
        int octant

    if abs_x < m.ROOT4_DBL_EPSILON:
        x2 = x * x
        res.val = x * (1 - x * x / 6.)
        res.err = cm.fabs(x * x2 * x2 / 100.0)
    else:
        y = cm.floor(abs_x / (0.25 * PI))
        octant = <int>(y - cm.ldexp(cm.floor(cm.ldexp(y, -3)), 3))

        if octant % 2 == 1:
            octant += 1
            octant &= 7
            y += 1

        if octant > 3:
            octant -= 4
            sgn *= -1

        z = abs_x - y * 7.85398125648498535156e-1 - y * 3.77489470793079817668e-8 - y * 2.69515142907905952645e-15

        t = 8 * cm.fabs(z) / PI - 1
        if octant == 0:
            res.val = z * (1 + z * z * cheb_eval(sin_constants, t, -1, 1).val)
        else:
            res.val = 1 - 0.5 * z * z * (1 - z * z * cheb_eval(cos_constants, t, -1, 1).val)

        res.val *= sgn

        z = cm.fabs(res.val)
        if abs_x > 1/m.DBL_EPSILON:
            res.err = z
        elif abs_x > 100/m.SQRT_DBL_EPSILON:
            res.err = 2.0 * abs_x * m.DBL_EPSILON * z
        elif abs_x > 0.1/m.SQRT_DBL_EPSILON:
            res.err = 2.0 * m.SQRT_DBL_EPSILON * z
        else:
            res.err = 2.0 * m.DBL_EPSILON * z

    res.err += cm.fabs(cm.cos(x) * dx) + m.DBL_EPSILON * res.val

    return res
