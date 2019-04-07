from .._specfunc import airy as a
from .._specfunc import airy_deriv as d


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
    return a.airy_Ai(x)


def airy_Ai_deriv(x):
    """
    Compute the derivative of the Airy function the first kind

    Parameters
    ----------
    x: {array_like, scalar}
        Numerical vector

    Returns
    -------
    array_like or scalar
        Derivative values from the Airy function
    """
    return d.airy_Ai_deriv(x)


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
    return a.airy_Ai_scaled(x)


def airy_Ai_deriv_scaled(x):
    """
    Compute the scaled derivative of the Airy function the first kind

    Parameters
    ----------
    x: {array_like, scalar}
        Numerical vector

    Returns
    -------
    array_like or scalar
        Derivative values from the Airy function
    """
    return d.airy_Ai_deriv_scaled(x)


def airy_Bi(x):
    r"""
    Computes the Airy function of the second kind. This is defined as

    .. math::

        Bi(x) = (1/\pi) \int_0^\infty \left[ e^{-(t^3/3) + xt} + \sin((t^3/3) + xt) \right] dt

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
    return a.airy_Bi(x)


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
    return a.airy_Bi_scaled(x)
