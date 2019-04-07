from ._results cimport Result

cdef:
    double cheb_eval(double[::1], double) nogil
    double cheb_eval_(double*, double, int) nogil
    double cheb_eval_mode(double[::1], double, int, int) nogil
    Result cheb_eval_mode_e(double[::1], double, int, int) nogil
