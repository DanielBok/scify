cdef Result make_r(double val, double err) nogil:
    cdef Result r
    r.val = val
    r.err = err
    return r

cdef ComplexResult make_c(double real, double real_err, double imag, double imag_err) nogil:
    cdef ComplexResult c
    c.real = real
    c.real_err = real_err
    c.imag = imag
    c.imag_err = imag
    return c
