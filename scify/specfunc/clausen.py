from .._specfunc import clausen as c


def clausen(x, threaded=True):
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

    threaded: bool, optional
        If True, uses multi-threading. Multi-threading is supported by the OpenMP api.

    Returns
    -------
    array_like or scalar
        Clausen output
    """
    return c.clausen(x, threaded)
