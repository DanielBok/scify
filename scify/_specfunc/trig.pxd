from ._results cimport Result

cdef:
    Result angle_restrict_pos_err(double) nogil
    Result cos_err(const double, const double) nogil
    Result sin_err(const double, const double) nogil
