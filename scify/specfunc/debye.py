from .._specfunc import debye as d

__all__ = ['debye_n', 'debye_1', 'debye_2', 'debye_3', 'debye_4', 'debye_5', 'debye_6']


def debye_n(x, order=1):
    r"""
    Computes the nth order Debye function

    .. math::

        D_n(x) = n/x^n \int^x_0 (t^n/(e^t - 1)) dt

    where :math:`n` represents the order

    Parameters
    ----------
    x: array_like
        Real values

    order: {1, 2, 3, 4, 5, 6}
        Order of the Debye function. Limited to 6.

    Returns
    -------
    {scalar or ndarray}
        Value of the Debye function
    """
    assert 1 <= order <= 6, "Debye order must be between [1, 6]"
    if order == 1:
        return debye_1(x)
    elif order == 2:
        return debye_2(x)
    elif order == 3:
        return debye_3(x)
    elif order == 4:
        return debye_4(x)
    elif order == 5:  # pragma: no cover
        return debye_5(x)
    elif order == 6:  # pragma: no cover
        return debye_6(x)


def debye_1(x):
    r"""
    Computes the first-order Debye function

    .. math::

        D_1(x) = \frac{1}{x} \int^x_0 \frac{t}{e^t - 1} dt

    Parameters
    ----------
    x: array_like
        Real values

    Returns
    -------
    {scalar or ndarray}
        Value of the Debye function
    """
    return d.debye_1(x)


def debye_2(x):
    r"""
    Computes the second-order Debye function

    .. math::
        D_2(x) = \frac{2}{x^2} \int^x_0 \frac{t^2}{e^t - 1} dt

    Parameters
    ----------
    x: array_like
        Real values

    Returns
    -------
    {scalar or ndarray}
        Value of the Debye function
    """
    return d.debye_2(x)


def debye_3(x):
    r"""
    Computes the third-order Debye function

    .. math::

        D_3(x) = \frac{3}{x^3} \int^x_0 \frac{t^3}{e^t - 1} dt

    Parameters
    ----------
    x: array_like
        Real values

    Returns
    -------
    {scalar or ndarray}
        Value of the Debye function
    """
    return d.debye_3(x)


def debye_4(x):
    r"""
    Computes the fourth-order Debye function

    .. math::

        D_4(x) = \frac{4}{x^4} \int^x_0 \frac{t^4}{e^t - 1} dt

    Parameters
    ----------
    x: array_like
        Real values

    Returns
    -------
    {scalar or ndarray}
        Value of the Debye function
    """
    return d.debye_4(x)


def debye_5(x):  # pragma: no cover
    r"""
    Computes the fifth-order Debye function

    .. math::

        D_5(x) = \frac{5}{x^5} \int^x_0 \frac{t^5}{e^t - 1} dt

    Parameters
    ----------
    x: array_like
        Real values

    Returns
    -------
    {scalar or ndarray}
        Value of the Debye function
    """
    return d.debye_5(x)


def debye_6(x):  # pragma: no cover
    r"""
    Computes the sixth-order Debye function

    .. math::

        D_6(x) = \frac{6}{x^6} \int^x_0 \frac{t^6}{e^t - 1} dt

    Parameters
    ----------
    x: array_like
        Real values

    Returns
    -------
    {scalar or ndarray}
        Value of the Debye function
    """
    return d.debye_6(x)
