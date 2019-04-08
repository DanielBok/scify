ctypedef struct Result:
    double val
    double err

ctypedef struct ComplexResult:
    double real
    double real_err
    double imag
    double imag_err


ctypedef Result (*Fn1R) (double) nogil
ctypedef ComplexResult (*Fn1C) (double, double) nogil


cdef:
    Result make_r(double val, double err) nogil
    Result make_r_0() nogil
    Result make_r_nan() nogil
    void map_dbl_p(Fn1R, double[::1], int) nogil
    void map_dbl_s(Fn1R, double[::1], int) nogil

    ComplexResult make_c(double real, double real_err, double imag, double imag_err) nogil
    ComplexResult make_c_0() nogil
    ComplexResult make_c_nan() nogil
    void mapc_dbl_p(Fn1C, double[::1], double[::1], int) nogil
    void mapc_dbl_s(Fn1C, double[::1], double[::1], int) nogil
