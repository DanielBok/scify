from .._specfunc import airy as a


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
        Values from the Airy function
    """
    return a.airy_Ai(x)


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
    return a.airy_Ai_scaled(x)
