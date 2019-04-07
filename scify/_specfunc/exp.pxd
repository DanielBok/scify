from ._results cimport Result

cdef:
    Result exp_mult_err(double, double, double, double) nogil
