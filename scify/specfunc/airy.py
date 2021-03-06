from scify.types import Real
from .._specfunc import airy as a
from .._specfunc import airy_deriv as d
from .._specfunc import airy_zero as z

__all__ = ['airy_Ai', 'airy_Ai_scaled', 'airy_Ai_deriv', 'airy_Ai_deriv_scaled', 'airy_zero_Ai', 'airy_zero_Ai_deriv',
           'airy_Bi', 'airy_Bi_scaled', 'airy_Bi_deriv', 'airy_Bi_deriv_scaled', 'airy_zero_Bi', 'airy_zero_Bi_deriv']


def airy_Ai(x, threaded=True) -> Real:
    r"""
    Computes the Airy function of the first kind. This is defined as

    .. math::

        Ai(x) = (1/\pi) \int_0^\infty \cos(\t^3/3 + xt) dt

    For more information, checkout the article on `Wikipedia <https://en.wikipedia.org/wiki/Airy_function>`_

    Parameters
    ----------
    x: array_like
        Numerical vector

    threaded: bool, optional
        If True, uses multi-threading. Multi-threading is supported by the OpenMP api.

    Returns
    -------
    array_like or scalar
        Values from the Airy function
    """
    return a.airy_Ai(x, threaded)


def airy_Ai_deriv(x, threaded=True) -> Real:
    """
    Compute the derivative of the Airy function the first kind

    Parameters
    ----------
    x: array_like
        Numerical vector

    threaded: bool, optional
        If True, uses multi-threading. Multi-threading is supported by the OpenMP api.

    Returns
    -------
    array_like or scalar
        Derivative values from the Airy function
    """
    return d.airy_Ai_deriv(x, threaded)


def airy_Ai_scaled(x, threaded=True) -> Real:
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
    x: array_like
        Numerical vector

    threaded: bool, optional
        If True, uses multi-threading. Multi-threading is supported by the OpenMP api.

    Returns
    -------
    array_like or scalar
        Values from the Airy function
    """
    return a.airy_Ai_scaled(x, threaded)


def airy_Ai_deriv_scaled(x, threaded=True) -> Real:
    """
    Compute the scaled derivative of the Airy function the first kind

    Parameters
    ----------
    x: array_like
        Numerical vector

    threaded: bool, optional
        If True, uses multi-threading. Multi-threading is supported by the OpenMP api.

    Returns
    -------
    array_like or scalar
        Derivative values from the Airy function
    """
    return d.airy_Ai_deriv_scaled(x, threaded)


def airy_zero_Ai(x, threaded=True) -> Real:
    r"""
    Compute the location of the s-th zero of the Airy function :math:`Ai(x)`

    Parameters
    ----------
    x: array_like
        Integer valued scalar or vector

    threaded: bool, optional
        If True, uses multi-threading. Multi-threading is supported by the OpenMP api.

    Returns
    -------
    array_like or scalar
        Location of the s-th zero of the Airy function
    """
    return z.airy_zero_Ai(x, threaded)


def airy_zero_Ai_deriv(x, threaded=True) -> Real:
    r"""
    Compute the location of the s-th zero of the Airy function derivative :math:`Ai'(x)`.

    Parameters
    ----------
    x: array_like
        Integer valued scalar or vector

    threaded: bool, optional
        If True, uses multi-threading. Multi-threading is supported by the OpenMP api.

    Returns
    -------
    array_like or scalar
        Location of the s-th zero of the Airy function derivative
    """
    return z.airy_zero_Ai_deriv(x, threaded)


def airy_Bi(x, threaded=True) -> Real:
    r"""
    Computes the Airy function of the second kind. This is defined as

    .. math::

        Bi(x) = (1/\pi) \int_0^\infty \left[ e^{-(t^3/3) + xt} + \sin((t^3/3) + xt) \right] dt

    For more information, checkout the article on `Wikipedia <https://en.wikipedia.org/wiki/Airy_function>`_

    Parameters
    ----------
    x: array_like
        Numerical vector

    threaded: bool, optional
        If True, uses multi-threading. Multi-threading is supported by the OpenMP api.

    Returns
    -------
    array_like or scalar
        Values from the Airy function
    """
    return a.airy_Bi(x, threaded)


def airy_Bi_deriv(x, threaded=True) -> Real:
    r"""
    Compute the derivative of the Airy function the second kind.

    Parameters
    ----------
    x: array_like
        Numerical vector

    threaded: bool, optional
        If True, uses multi-threading. Multi-threading is supported by the OpenMP api.

    Returns
    -------
    array_like or scalar
        Derivative values from the Airy function
    """
    return d.airy_Bi_deriv(x, threaded)


def airy_Bi_scaled(x, threaded=True) -> Real:
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
    x: array_like
        Numerical vector

    threaded: bool, optional
        If True, uses multi-threading. Multi-threading is supported by the OpenMP api.

    Returns
    -------
    array_like or scalar
        Values from the Airy function
    """
    return a.airy_Bi_scaled(x, threaded)


def airy_Bi_deriv_scaled(x, threaded=True) -> Real:
    r"""
    Compute the scaled derivative of the Airy function the second kind.

    Parameters
    ----------
    x: array_like
        Numerical vector

    threaded: bool, optional
        If True, uses multi-threading. Multi-threading is supported by the OpenMP api.

    Returns
    -------
    array_like or scalar
        Derivative values from the Airy function
    """
    return d.airy_Bi_deriv_scaled(x, threaded)


def airy_zero_Bi(x, threaded=True) -> Real:
    r"""
    Compute the location of the s-th zero of the Airy function :math:`Bi(x)`

    Parameters
    ----------
    x: array_like
        Integer valued scalar or vector

    threaded: bool, optional
        If True, uses multi-threading. Multi-threading is supported by the OpenMP api.

    Returns
    -------
    array_like or scalar
        Location of the s-th zero of the Airy function
    """
    return z.airy_zero_Bi(x, threaded)


def airy_zero_Bi_deriv(x, threaded=True) -> Real:
    r"""
    Compute the location of the s-th zero of the Airy function derivative :math:`Bi'(x)`.

    Parameters
    ----------
    x: array_like
        Integer valued scalar or vector

    threaded: bool, optional
        If True, uses multi-threading. Multi-threading is supported by the OpenMP api.

    Returns
    -------
    array_like or scalar
        Location of the s-th zero of the Airy function derivative
    """
    return z.airy_zero_Bi_deriv(x, threaded)
