# Cython compiler directives.
# distutils: language=c++
# cython: profile=False


import numpy as np
import os
import pykonal
import scipy.optimize
import tempfile

from . import constants as _constants
from . import inventory as _inventory
from . import solver as _solver
from . import transformations as _transformations

cimport numpy as np

from libc.math cimport sqrt, isinf, isnan, INFINITY

from . cimport fields
from . cimport constants


cdef class EQLocator(object):
    """
    EQLocator(stations, tt_inv, coord_sys='spherical')
    
    A class to locate earthquakes.
    """
    def __init__(
        self,
        traveltime_inventory: str,
        coord_sys: str="spherical"
    ):
        self.cy_arrivals = {}
        self.cy_traveltimes = {}
        self.cy_residual_rvs = {}
        self.cy_coord_sys = coord_sys
        # new for adding uncertainty
        self.cy_sigma_pick    = 0.02
        self.cy_alpha         = 0.01

        inventory = _inventory.TraveltimeInventory(traveltime_inventory, mode="r")
        self.cy_traveltime_inventory = inventory


    def __del__(self):
        self.traveltime_inventory.f5.close()


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__del__()


    cpdef constants.BOOL_t add_arrivals(EQLocator self, dict arrivals):
        self.cy_arrivals = {**self.cy_arrivals, **arrivals}
        return True


    cpdef constants.BOOL_t add_residual_rvs(EQLocator self, dict residual_rvs):
        self.cy_residual_rvs = {**self.cy_residual_rvs, **residual_rvs}
        return True


    cpdef constants.BOOL_t clear_arrivals(EQLocator self):
        self.cy_arrivals = {}
        return True


    cpdef constants.BOOL_t clear_residual_rvs(EQLocator self):
        self.cy_residual_rvs = {}
        return True
        
    
    @property
    def arrivals(self) -> dict:
        return self.cy_arrivals
    
    @arrivals.setter
    def arrivals(self, value: dict):
        self.cy_arrivals = value

    @property
    def coord_sys(self) -> str:
        return self.cy_coord_sys

    @property
    def grid(self) -> object:
        if self.cy_grid is None:
            self.cy_grid = fields.ScalarField3D(coord_sys=self.cy_coord_sys)
        return self.cy_grid

    @property
    def traveltime_inventory(self) -> object:
        return self.cy_traveltime_inventory

    @property
    def pwave_velocity(self) -> object:
        if self.cy_pwave_velocity is None:
            self.cy_pwave_velocity = fields.ScalarField3D(
                coord_sys=self.cy_coord_sys
            )
            self.cy_pwave_velocity.min_coords = self.cy_grid.min_coords
            self.cy_pwave_velocity.node_intervals = self.cy_grid.node_intervals
            self.cy_pwave_velocity.npts = self.cy_grid.npts
        return self.cy_pwave_velocity
    
    @pwave_velocity.setter
    def pwave_velocity(self, value: np.ndarray):
        if self.cy_pwave_velocity is None:
            self.pwave_velocity
        self.cy_pwave_velocity.values = value
    
    @property
    def vp(self) -> object:
        return self.pwave_velocity
    
    @vp.setter
    def vp(self, value: np.ndarray):
        self.pwave_velocity = value

    @property
    def residual_rvs(self) -> dict:
        return self.cy_residual_rvs
    
    @residual_rvs.setter
    def residual_rvs(self, value: dict):
        self.cy_residual_rvs = value
    
    @property
    def swave_velocity(self) -> object:
        if self.cy_swave_velocity is None:
            self.cy_swave_velocity = fields.ScalarField3D(
                coord_sys=self.cy_coord_sys
            )
            self.cy_swave_velocity.min_coords = self.cy_grid.min_coords
            self.cy_swave_velocity.node_intervals = self.cy_grid.node_intervals
            self.cy_swave_velocity.npts = self.cy_grid.npts
        return self.cy_swave_velocity
    
    @swave_velocity.setter
    def swave_velocity(self, value: np.ndarray):
        if self.cy_swave_velocity is None:
            self.swave_velocity
        self.cy_swave_velocity.values = value
        
    @property
    def traveltimes(self) -> dict:
        return self.cy_traveltimes
    
    @traveltimes.setter
    def traveltimes(self, value: dict):
        self.cy_traveltimes = value
        
    @property
    def vs(self) -> object:
        return self.swave_velocity

    @vs.setter
    def vs(self, value: np.ndarray):
        self.swave_velocity = value

    @property
    def sigma_pick(self):
        return self.cy_sigma_pick

    @sigma_pick.setter
    def sigma_pick(self, value):
        if value < 0:
            raise ValueError("sigma_pick must be >= 0")
        self.cy_sigma_pick = value

    @property
    def alpha(self):
        return self.cy_alpha

    @alpha.setter
    def alpha(self, value):
        if value < 0:
            raise ValueError("alpha must be >= 0")
        self.cy_alpha = value

    cpdef constants.BOOL_t read_traveltimes(
        EQLocator self, 
        constants.REAL_t[:] min_coords=None, 
        constants.REAL_t[:] max_coords=None
    ):

        inventory = self.cy_traveltime_inventory
        self.cy_traveltimes = {
            index: inventory.read(
                "/".join(index),
                min_coords=min_coords,
                max_coords=max_coords
            ) for index in self.cy_arrivals
        }

        return True

    """
    cpdef constants.REAL_t rms(EQLocator self, constants.REAL_t[:] hypocenter):
        cdef tuple key
        cdef dict arrivals = self.cy_arrivals
        cdef dict traveltimes = self.cy_traveltimes
        cdef constants.REAL_t csum = 0
        cdef constants.REAL_t num
        cdef constants.REAL_t tt
        cdef constants.REAL_t arrival_time
        cdef constants.REAL_t t0 = hypocenter[3]
        cdef constants.REAL_t[:] hypo_xyz = hypocenter[:3]
        cdef int valid_measurements = 0
        cdef fields.ScalarField3D tt_field

        for key, arrival_time in arrivals.items():
            tt_field = traveltimes[key]
            tt = tt_field.value(hypo_xyz, null=INFINITY)
            if isnan(tt) or tt > 9999:
                continue
            num = arrival_time - t0 - tt
            csum += num * num
            valid_measurements += 1

        if valid_measurements == 0:
            return 1e6

        return sqrt(csum / valid_measurements)
    """

    # weighted RMS (new)
    cpdef constants.REAL_t rms(EQLocator self, constants.REAL_t[:] hypocenter):
        cdef tuple key
        cdef dict arrivals = self.cy_arrivals
        cdef dict traveltimes = self.cy_traveltimes
        cdef constants.REAL_t csum = 0
        cdef constants.REAL_t weight_sum = 0
        cdef constants.REAL_t num
        cdef constants.REAL_t tt
        cdef constants.REAL_t variance
        cdef constants.REAL_t weight
        cdef constants.REAL_t t0 = hypocenter[3]
        cdef constants.REAL_t[:] hypo_xyz = hypocenter[:3]
        cdef constants.REAL_t sigma_pick_sq = self.cy_sigma_pick * self.cy_sigma_pick
        cdef constants.REAL_t alpha_sq = self.cy_alpha * self.cy_alpha
        cdef int valid_measurements = 0

        for key in arrivals:
            tt = traveltimes[key].value(hypo_xyz, null=INFINITY)
            if isinf(tt) or isnan(tt) or tt > 9999:
                continue
            num = arrivals[key] - t0 - tt
            variance = sigma_pick_sq + alpha_sq * tt * tt
            weight = 1.0 / variance
            csum += weight * num * num
            weight_sum += weight
            valid_measurements += 1

        if valid_measurements == 0:
            return 1e6

        return sqrt(csum / weight_sum)


    cpdef np.ndarray[constants.REAL_t, ndim=1] locate(
        EQLocator self,
        np.ndarray[constants.REAL_t, ndim=1] initial,
        np.ndarray[constants.REAL_t, ndim=1] delta,
        constants.REAL_t sigma_pick=0.02, 
        constants.REAL_t alpha=0.01        
    ):
        """
        Locate event using a grid search and Differential Evolution
        Optimization to minimize the residual RMS.
        sigma_pick (seconds) picking precision floor
        alpha = fractional traveltime error in % 
        """
        self.cy_sigma_pick = sigma_pick
        self.cy_alpha = alpha
      
        min_coords = initial - delta
        max_coords = initial + delta

        bounds = np.stack([min_coords, max_coords]).T

        self.read_traveltimes(
            min_coords=min_coords[:3],
            max_coords=max_coords[:3]
        )

        # RCP added some kwargs
        soln = scipy.optimize.differential_evolution(self.rms, bounds,
                                                     x0 = initial, # recent scipy allows an initial estimate 
                                                     strategy='best1bin', updating='immediate', 
                                                     maxiter=200, mutation=(0.3,1.0), recombination=0.7,
                                                     popsize=20, atol=0.01, tol=0.01, init='sobol',
                                                     polish=False)
        #soln = scipy.optimize.differential_evolution(self.rms, bounds, strategy='best1bin') # original

        # Polish (find the bottom of the basin)
        polished = scipy.optimize.minimize(
            self.rms, soln.x,
            method='Nelder-Mead',
            options={
                'xatol': 0.05,    # 50 m / 50 ms — tighter than DE could give
                'fatol': 0.005,   # 5 ms RMS resolution
                'maxiter': 100,
            },
        )

        final_x = polished.x if polished.fun < soln.fun else soln.x

        # so the solution is the minium rms in the DE cloud (shape (4,) x,y,z,t)
        # we could at some point accept/reject based on this - send to pyvorotomo to boot events
        # final_rms = min(polished.fun, soln.fun)
        # soln_std = np.std(soln.population, axis=0) # n.b. this is pre-polish
        # return final_x,final_rms,soln_std

        return final_x


    cpdef constants.REAL_t log_likelihood(
        EQLocator self,
        constants.REAL_t[:] model
    ):
        cdef constants.REAL_t   t_pred, residual
        cdef constants.REAL_t   log_likelihood = 0.0
        cdef tuple              key

        for key in self.cy_arrivals:
            t_pred = model[3] + self.cy_traveltimes[key].value(model[:3])
            residual = self.cy_arrivals[key] - t_pred
            log_likelihood = log_likelihood + self.cy_residual_rvs[key].logpdf(residual)
        return (log_likelihood)
