from .._specfunc import clausen as c


def clausen(x):
    r"""
    The Clausen function is defined by the following integral,

    .. math::

        Cl_2(x) = - \int_0^x \log(2 \sin(t/2)) dt

    See the `Wikipedia <https://en.wikipedia.org/wiki/Clausen_function>`_ article
    for more information.

    Parameters
    ----------
    x: {array_like, scalar}
        Numeric vector input

    Returns
    -------
    {array_like, scalar}
        Clausen output
    """
    return c.clausen(x)
