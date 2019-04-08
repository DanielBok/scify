from ._results cimport ComplexResult, Result

cdef:
    ComplexResult _complex_log(double, double) nogil
    Result log_1plusx(double x) nogil
    Result log_1plusx_mx(double x) nogil
