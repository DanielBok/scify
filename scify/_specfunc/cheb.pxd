from ._results cimport Result

cdef:
    Result cheb_eval(double[::1], double, int, int) nogil
    Result cheb_eval_mode(double[::1], double, int, int) nogil
