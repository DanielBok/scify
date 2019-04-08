from ._results cimport ComplexResult

cdef:
    ComplexResult _complex_log(double, double) nogil
