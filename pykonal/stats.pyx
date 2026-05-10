cimport numpy as np

from . cimport constants

from libc.math cimport log, pi, sqrt

cdef double NEG_HALF_LOG_2PI = -0.5 * log(2 * pi)

cdef class NormalDistribution(object):

    def __init__(self, mu: constants.REAL_t=0, sigma: constants.REAL_t=1):
        self.cy_mu = mu
        self.cy_sigma = sigma
        self.cy_log_sigma = log(sigma)
        self.cy_inv_sigma = 1.0 / sigma

    @property
    def mu(self):
        return self.cy_mu

    @property
    def sigma(self):
        return self.cy_sigma

    cpdef constants.REAL_t logpdf(NormalDistribution self, constants.REAL_t x):
        cdef constants.REAL_t z
        z = (x - self.cy_mu) * self.cy_inv_sigma
        return (NEG_HALF_LOG_2PI - self.cy_log_sigma - 0.5 * z * z)
