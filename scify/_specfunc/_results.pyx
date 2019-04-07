from cython.parallel import prange
from libc.math cimport NAN

cdef Result make_r(double val, double err) nogil:
    cdef Result r
    r.val = val
    r.err = err
    return r


cdef Result make_r_0() nogil:
    return make_r(0, 0)


cdef Result make_r_nan() nogil:
    return make_r(NAN, NAN)


cdef ComplexResult make_c(double real, double real_err, double imag, double imag_err) nogil:
    cdef ComplexResult c
    c.real = real
    c.real_err = real_err
    c.imag = imag
    c.imag_err = imag
    return c


cdef ComplexResult make_c_0() nogil:
    return make_c(0, 0, 0, 0)


cdef ComplexResult make_c_nan() nogil:
    return make_c(NAN, NAN, NAN, NAN)


cdef void map_dbl_p(Fn1R f, double[::1] x, int size) nogil:
    """Parallel"""
    cdef int i
    for i in prange(size, nogil=True):
        x[i] = f(x[i]).val


cdef void map_dbl_s(Fn1R f, double[::1] x, int size) nogil:
    """Single Thread"""
    cdef int i
    for i in range(size):
        x[i] = f(x[i]).val


cdef void mapc_dbl_p(Fn1C f, double[::1] r, double[::1] t, int size) nogil:
    """Parallel"""
    cdef:
        ComplexResult c
        int i

    for i in prange(size, nogil=True):
        c = f(r[i], t[i])
        r[i] = c.real
        t[i] = c.imag


cdef void mapc_dbl_s(Fn1C f, double[::1] r, double[::1] t, int size) nogil:
    """Single Thread"""
    cdef:
        ComplexResult c
        int i

    for i in range(size):
        c = f(r[i], t[i])
        r[i] = c.real
        t[i] = c.imag
