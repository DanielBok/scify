ctypedef struct Result:
    double val
    double err

ctypedef struct ComplexResult:
    double real
    double real_err
    double imag
    double imag_err

cdef:
    Result make_r(double val, double err) nogil
    ComplexResult make_c(double real, double real_err, double imag, double imag_err) nogil
