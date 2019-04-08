import numpy as np

from scify.types import Complex
from .._specfunc import log as sl

__all__ = ['complex_log']


def complex_log(zr: Complex, zi=None, threaded=True) -> Complex:
    r"""
    Function returns the complex natural logarithm (base e) of the complex number z, :math:`\log(z)`.

    The branch cut is the negative real axis.

    Parameters
    ----------
    zr: array_like
        The real component of the complex vector or scalar. If `zi` is None, interpret `zr` as a complex valued object

    zi: array_like, optional
        The imaginary component of the complex vector or scalar.

    threaded: bool, optional
        If True, uses multi-threading. Multi-threading is supported by the OpenMP api.

    Returns
    -------
    array_like or scalar
        Complex Dilog output
    """
    if zi is None:
        zi = np.imag(zr)
        zr = np.real(zr)

    return sl.complex_log(zr, zi, threaded)
