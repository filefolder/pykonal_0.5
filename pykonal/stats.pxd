from . cimport constants


cdef class NormalDistribution(object):
    cdef constants.REAL_t cy_mu
    cdef constants.REAL_t cy_sigma
    cdef constants.REAL_t cy_log_sigma
    cdef constants.REAL_t cy_inv_sigma
    
    cpdef constants.REAL_t logpdf(NormalDistribution self, constants.REAL_t x)
