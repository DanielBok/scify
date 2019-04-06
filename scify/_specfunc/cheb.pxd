cdef:
    double cheb_eval(double*, double, int) nogil
    double cheb_eval_mode(double* constants, double x, int order, int a, int b) nogil
