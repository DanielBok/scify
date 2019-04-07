from ._results cimport Result

cdef:
    double cheb_eval(double[:], double) nogil
    double cheb_eval_(double*, double, int) nogil
    double cheb_eval_mode(double[:], double, int, int) nogil
    Result cheb_eval_mode_e(double[:], double, int, int) nogil
