import numpy as np

from .._specfunc import dilog as d


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
    array_like or scalar
        Real Dilog output
    """
    return d.dilog(x)


def dilog_complex(r, theta=None):
    r"""
    This function computes the full complex-valued dilogarithm for the complex argument
    :math=:`z = r \exp(i \theta)`.

    Parameters
    ----------
    r: {array_like, scalar}
        The modulus of the complex vector or scalar. If `theta` is None, interpret `r` as a complex valued object
    theta: {array_like, scalar}, optional
        The argument of the complex vector or scalar

    Returns
    -------
    array_like or scalar
        Complex Dilog output
    """
    if theta is None:
        theta = np.angle(r)
        r = np.abs(r)

    return d.dilog_complex(r, theta)
