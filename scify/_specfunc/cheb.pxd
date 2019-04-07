cdef:
    double cheb_eval(double[:], double) nogil
    double cheb_eval_(double*, double, int) nogil
    double cheb_eval_mode(double[:], double, int, int) nogil
    double cheb_eval_mode_(double*, double, int, int, int) nogil
