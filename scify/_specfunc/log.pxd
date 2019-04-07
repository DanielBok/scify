from ._results cimport ComplexResult

cdef:
    ComplexResult complex_log(double, double) nogil
