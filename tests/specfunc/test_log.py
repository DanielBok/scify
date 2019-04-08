import numpy as np
import pytest
from numpy.testing import assert_almost_equal

import scify.specfunc.log as lg


@pytest.mark.parametrize("z, exp", [
    (-1.5 - 3.14159265358979j, 1.24741216996716 - 2.01625382070549j),
    (0 - 3.14159265358979j, 1.1447298858494 - 1.5707963267949j),
    (2 - 3.14159265358979j, 1.31484985600431 - 1.00388482185389j),
    (-1.5 + 0j, 0.40546510810816 + 3.14159265358979j),
    (0 + 0j, np.nan + np.nan * 1j),
    (2 + 0j, 0.693147180559945 + 0j),
    (-1.5 + 2.71828182845905j, 1.13291159438989 + 2.07503513405872j),
    (0 + 2.71828182845905j, 1 + 1.5707963267949j),
    (2 + 2.71828182845905j, 1.2163264514959 + 0.93647200756652j),
    [
        (
                -1.5 - 3.14159265358979j, 0 - 3.14159265358979j, 2 - 3.14159265358979j, -1.5 + 0j, 0 + 0j, 2 + 0j,
                -1.5 + 2.71828182845905j, 0 + 2.71828182845905j, 2 + 2.71828182845905j
        ),
        (
                1.24741216996716 - 2.01625382070549j, 1.1447298858494 - 1.5707963267949j,
                1.31484985600431 - 1.00388482185389j, 0.40546510810816 + 3.14159265358979j, np.nan + np.nan * 1j,
                0.693147180559945 + 0j, 1.13291159438989 + 2.07503513405872j, 1 + 1.5707963267949j,
                1.2163264514959 + 0.93647200756652j
        )
    ]
])
def test_complex_log(z, exp):
    numbers = lg.complex_log(z)

    if isinstance(z, (complex, float, int)):
        re = numbers.real
        im = numbers.imag

        t_re = exp.real
        t_im = exp.imag
    else:
        re = np.real(numbers)
        im = np.imag(numbers)

        t_re = np.real(exp)
        t_im = np.imag(exp)

    assert_almost_equal(re, t_re)
    assert_almost_equal(im, t_im)


def test_benchmark_complex_log(benchmark, complex_data):
    benchmark(lg.complex_log, complex_data, threaded=False)
