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

from libc.math cimport sqrt, isinf, isnan, INFINITY, NAN, exp, log, atan2, M_PI, fmod, sin, cos, asin

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
        # new for down-weighting distant traveltimes
        self.cy_alpha         = 0.01
        # NLL-style additions
        self.cy_stations            = {}
        self.cy_pick_errors         = {}
        self.cy_default_pick_error  = 0.02  # seconds
        self.cy_edt_exponent        = 1.0
        # Quadratic regularization weight blended into the EDT log-
        # likelihood. Adding a small quadratic differential-time misfit
        # restores curvature along EDT's otherwise-flat ridges, which on
        # WELL-CONSTRAINED (azimuthally surrounded) events sharply improves
        # both accuracy and seed/start stability (benchmarks: horizontal
        # scatter ~26 km -> ~6 km under +/-3 s pick noise and 10 km start
        # shifts). HOWEVER, on WEAK one-sided geometry the L2 minimum sits
        # in a false far-field basin, so regularization pulls the solution
        # far off (benchmarks: ~23 km -> ~120 km error). It is therefore
        # OPT-IN and defaults to 0 (pure NLL-style EDT). Enable it (e.g.
        # 0.05-0.2) only for networks whose events are well surrounded.
        self.cy_edt_reg             = 0.0
        # Differential-evolution search controls for the EDT locator.
        # Larger popsize/maxiter make the global search more robust on
        # rough/ridged surfaces at the cost of runtime; tol is the DE
        # convergence tolerance. Exposed as properties for tuning.
        # Fixed default seed makes the DE search deterministic: the same
        # input yields the same solution across repeated calls / processes
        # (e.g. scolv relocating twice). Set locate_seed=None for the old
        # nondeterministic behavior.
        self.cy_locate_seed         = 12345
        self.cy_keys                = None
        self.cy_tt_fields           = None

        self.cy_traveltime_inventory = None
        inventory = _inventory.TraveltimeInventory(traveltime_inventory, mode="r")
        self.cy_traveltime_inventory = inventory


    def __del__(self):
        if self.cy_traveltime_inventory is not None:
            self.traveltime_inventory.f5.close()
            # Prevent a second close when garbage collection runs
            # __del__ again after an explicit close via __exit__.
            self.cy_traveltime_inventory = None


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__del__()


    cpdef constants.BOOL_t add_arrivals(EQLocator self, dict arrivals):
        self.cy_arrivals = {**self.cy_arrivals, **arrivals}
        self.cy_keys = None
        return True


    cpdef constants.BOOL_t add_residual_rvs(EQLocator self, dict residual_rvs):
        self.cy_residual_rvs = {**self.cy_residual_rvs, **residual_rvs}
        return True


    cpdef constants.BOOL_t add_stations(EQLocator self, dict stations):
        """
        Register station coordinates, e.g. {("NET", "STA"): (x, y, z)} in the
        same coordinate system as the traveltime grids. Used for quality
        metrics (azimuthal gap, station distances). Keys may also be
        3-tuples ("NET", "STA", "P"); the phase element is ignored.
        """
        cleaned = {}
        for key, value in stations.items():
            key = tuple(key)
            if len(key) == 3:
                key = key[:2]
            cleaned[key] = np.asarray(value, dtype=np.float64)
        self.cy_stations = {**self.cy_stations, **cleaned}
        return True


    cpdef constants.BOOL_t add_pick_errors(EQLocator self, dict pick_errors):
        """
        Per-arrival pick uncertainties (seconds), keyed like arrivals:
        {("NET", "STA", "P"): 0.05, ...}. Arrivals without an entry use
        self.default_pick_error.
        """
        self.cy_pick_errors = {**self.cy_pick_errors, **pick_errors}
        self.cy_keys = None
        return True


    cpdef constants.BOOL_t clear_arrivals(EQLocator self):
        self.cy_arrivals = {}
        self.cy_keys = None
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
    def alpha(self):
        return self.cy_alpha

    @alpha.setter
    def alpha(self, value):
        if value < 0:
            raise ValueError("alpha must be >= 0")
        self.cy_alpha = value

    @property
    def stations(self) -> dict:
        return self.cy_stations

    @stations.setter
    def stations(self, value: dict):
        self.cy_stations = {}
        self.add_stations(value)

    @property
    def pick_errors(self) -> dict:
        return self.cy_pick_errors

    @pick_errors.setter
    def pick_errors(self, value: dict):
        self.cy_pick_errors = value
        self.cy_keys = None

    @property
    def default_pick_error(self):
        return self.cy_default_pick_error

    @default_pick_error.setter
    def default_pick_error(self, value):
        if value <= 0:
            raise ValueError("default_pick_error must be > 0")
        self.cy_default_pick_error = value

    @property
    def edt_exponent(self):
        """
        Exponent applied to the EDT stack (NLL's PDF sharpening). 1.0 gives
        the classic EDT sum; larger values sharpen the posterior toward the
        best-fitting region as in NLL's EDT^N option.
        """
        return self.cy_edt_exponent

    @edt_exponent.setter
    def edt_exponent(self, value):
        if value <= 0:
            raise ValueError("edt_exponent must be > 0")
        self.cy_edt_exponent = value

    @property
    def edt_reg(self):
        """
        Quadratic-regularization weight added to the EDT log-likelihood to
        break the flat-ridge degeneracy on weak geometry. 0 disables it
        (pure EDT); larger values sharpen the minimum at some cost to
        outlier robustness. Default 0 (disabled); see the note in
        __init__ before enabling it on one-sided networks.
        """
        return self.cy_edt_reg

    @edt_reg.setter
    def edt_reg(self, value):
        if value < 0:
            raise ValueError("edt_reg must be >= 0")
        self.cy_edt_reg = value

    @property
    def locate_seed(self):
        """
        Random seed for the differential-evolution search in locate().
        An integer (default 12345) makes results reproducible; None
        restores nondeterministic behavior (fresh randomness each call).
        """
        return self.cy_locate_seed

    @locate_seed.setter
    def locate_seed(self, value):
        self.cy_locate_seed = value

    cpdef constants.BOOL_t read_traveltimes(
        EQLocator self, 
        constants.REAL_t[:] min_coords=None, 
        constants.REAL_t[:] max_coords=None
    ):

        inventory = self.cy_traveltime_inventory
        self.cy_traveltimes = {}
        dropped = []
        for index in self.cy_arrivals:
            key = "/".join(index)
            if not inventory.has(key):
                dropped.append(key)
                continue
            self.cy_traveltimes[index] = inventory.read(
                key,
                min_coords=min_coords,
                max_coords=max_coords
            )
        if dropped:
            import warnings
            warnings.warn(
                f"No traveltime grid for {len(dropped)} arrival(s): "
                f"{', '.join(dropped)}. These arrivals are excluded. "
                f"Use TraveltimeInventory.ensure() to compute missing "
                f"grids before locating.",
                stacklevel=2
            )

        # traveltime grids changed; flattened workspace is stale
        self.cy_keys = None
        self.cy_tt_fields = None

        return True


    cpdef constants.BOOL_t _prepare_workspace(EQLocator self):
        """
        Flatten arrivals/traveltimes/errors into aligned arrays
        so objective functions can run tight typed loops. Called lazily;
        invalidated whenever traveltimes are (re)read.
        """
        cdef int idx, n

        keys = sorted(set(self.cy_arrivals) & set(self.cy_traveltimes))
        n = len(keys)

        self.cy_keys      = keys
        self.cy_tt_fields = [self.cy_traveltimes[key] for key in keys]
        self.cy_obs       = np.array(
            [self.cy_arrivals[key] for key in keys], dtype=np.float64
        )
        self.cy_sigma     = np.array(
            [
                self.cy_pick_errors.get(key, self.cy_default_pick_error)
                for key in keys
            ],
            dtype=np.float64
        )
        self.cy_tt_work   = np.empty(n, dtype=np.float64)

        return True


    cdef int _fill_traveltimes(EQLocator self, constants.REAL_t[:] hypo_xyz):
        """
        Interpolate every arrival's traveltime at
        hypo_xyz into cy_tt_work. Invalid nodes are set to INFINITY. Returns
        the number of valid traveltimes.
        """
        cdef int idx, valid = 0
        cdef constants.REAL_t tt
        cdef fields.ScalarField3D tt_field

        if self.cy_keys is None:
            self._prepare_workspace()

        for idx in range(len(self.cy_keys)):
            tt_field = self.cy_tt_fields[idx]
            tt = tt_field.value(hypo_xyz, null=INFINITY)
            if isinf(tt) or isnan(tt) or tt > 9999:
                self.cy_tt_work[idx] = INFINITY
            else:
                self.cy_tt_work[idx] = tt
                valid += 1

        return valid


    cpdef constants.REAL_t edt_log_likelihood(
        EQLocator self,
        constants.REAL_t[:] hypo_xyz
    ):
        """
        NLL-style Equal Differential Time (EDT) log-likelihood at a trial
        hypocenter (3 spatial coordinates only; origin time cancels).

        For every pair of arrivals (a, b):

            r_ab = (t_obs_a - t_obs_b) - (tt_a - tt_b)
            L   += exp(-r_ab^2 / (s_a^2 + s_b^2)) / sqrt(s_a^2 + s_b^2)

        where s^2 = pick_error^2 + (alpha * tt)^2 combines pick uncertainty
        with a fractional traveltime (velocity-model) error, mirroring the
        role of alpha in the weighted-RMS objective. Returns
        edt_exponent * log(L / n_pairs); larger is better.
        """
        cdef int    ia, ib, n, npairs = 0
        cdef constants.REAL_t tta, ttb, r, va, vb, vv
        cdef constants.REAL_t stack = 0.0
        cdef constants.REAL_t reg_sum = 0.0
        cdef constants.REAL_t alpha_sq = self.cy_alpha * self.cy_alpha

        if self.cy_keys is None:
            self._prepare_workspace()

        n = len(self.cy_keys)
        if self._fill_traveltimes(hypo_xyz) < 2:
            return -INFINITY

        for ia in range(n - 1):
            tta = self.cy_tt_work[ia]
            if isinf(tta):
                continue
            va = self.cy_sigma[ia] * self.cy_sigma[ia] + alpha_sq * tta * tta
            for ib in range(ia + 1, n):
                ttb = self.cy_tt_work[ib]
                if isinf(ttb):
                    continue
                vb = self.cy_sigma[ib] * self.cy_sigma[ib] + alpha_sq * ttb * ttb
                vv = va + vb
                r = (self.cy_obs[ia] - self.cy_obs[ib]) - (tta - ttb)
                stack += exp(-(r * r) / vv) / sqrt(vv)
                reg_sum += (r * r) / vv     # quadratic differential misfit
                npairs += 1

        if npairs == 0 or stack <= 0.0:
            return -INFINITY

        # EDT log-likelihood (flat-topped, outlier-robust) minus a small
        # quadratic differential-time misfit that restores curvature along
        # otherwise-degenerate ridges. With cy_edt_reg = 0 this is exactly
        # the classic NLL EDT.
        return (self.cy_edt_exponent * log(stack / npairs)
                - self.cy_edt_reg * (reg_sum / npairs))


    cpdef constants.REAL_t edt(EQLocator self, constants.REAL_t[:] hypo_xyz):
        """
        Negative EDT log-likelihood: a misfit function suitable for
        minimization. Accepts either a 3-vector or a 4-vector (the origin
        time element, if present, is ignored since EDT is independent of it).
        """
        cdef constants.REAL_t ll = self.edt_log_likelihood(hypo_xyz[:3])
        if isinf(ll):
            return 1e6
        return -ll


    cpdef constants.REAL_t origin_time(EQLocator self, constants.REAL_t[:] hypo_xyz):
        """
        Origin time at a fixed hypocenter: the inverse-variance *weighted
        median* of (t_obs - tt). The median (rather than mean)
        keeps the decoupled origin-time estimate robust to the same outlier
        picks that EDT itself is immune to.
        """
        cdef int idx, n

        if self.cy_keys is None:
            self._prepare_workspace()

        n = len(self.cy_keys)
        if self._fill_traveltimes(hypo_xyz) == 0:
            return np.nan

        alpha_sq = self.cy_alpha * self.cy_alpha
        tt = np.asarray(self.cy_tt_work)
        valid = np.isfinite(tt)
        t0s = np.asarray(self.cy_obs)[valid] - tt[valid]
        sig = np.asarray(self.cy_sigma)[valid]
        w = 1.0 / (sig * sig + alpha_sq * tt[valid] * tt[valid])

        order = np.argsort(t0s)
        t0s, w = t0s[order], w[order]
        cw = np.cumsum(w)
        return float(t0s[np.searchsorted(cw, 0.5 * cw[-1])])

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
    """
    # weighted RMS (newer)
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
        cdef constants.REAL_t alpha_sq = self.cy_alpha * self.cy_alpha
        cdef int valid_measurements = 0

        for key in arrivals:
            tt = traveltimes[key].value(hypo_xyz, null=INFINITY)
            if isinf(tt) or isnan(tt) or tt > 9999:
                continue
            num = arrivals[key] - t0 - tt
            variance = alpha_sq * tt * tt
            weight = 1.0 / variance
            csum += weight * num * num
            weight_sum += weight
            valid_measurements += 1

        if valid_measurements == 0:
            return 1e6

        return sqrt(csum / weight_sum)
    """

    # L1 weighting, hopefully more immune to garbage
    # (now uses the flattened workspace)
    cpdef constants.REAL_t rms(EQLocator self, constants.REAL_t[:] hypocenter):
        cdef constants.REAL_t t0 = hypocenter[3]
        cdef constants.REAL_t[:] hypo_xyz = hypocenter[:3]
        cdef constants.REAL_t alpha_sq = self.cy_alpha * self.cy_alpha
        cdef constants.REAL_t tt, num, weight
        cdef int idx, n, valid_measurements = 0
        cdef constants.REAL_t csum = 0
        cdef constants.REAL_t weight_sum = 0

        if self.cy_keys is None:
            self._prepare_workspace()

        n = len(self.cy_keys)
        self._fill_traveltimes(hypo_xyz)

        for idx in range(n):
            tt = self.cy_tt_work[idx]
            if isinf(tt):
                continue
            num = self.cy_obs[idx] - t0 - tt
            weight = 1.0 / (1.0 + alpha_sq * tt * tt)
            csum += weight * (num if num >= 0 else -num)
            weight_sum += weight
            valid_measurements += 1

        if valid_measurements == 0:
            return 1e6

        return csum / weight_sum


    cpdef np.ndarray locate(
        EQLocator self,
        np.ndarray initial,
        np.ndarray delta,
        constants.REAL_t alpha=NAN,
        str method="l1"
    ):
        """
        Locate event using Differential Evolution followed by a Nelder-Mead
        polish.

        method="l1"  : minimize the distance-weighted L1 residual norm
                       (searches x, y, z, t0 jointly; original behavior).
        method="edt" : maximize the NLL-style EDT likelihood (searches
                       x, y, z only; origin time is recovered afterward
                       from the weighted mean residual). More robust to
                       outlier picks.

        alpha = fractional traveltime error (velocity-model uncertainty).
                If omitted, the locator's current self.alpha is used
                (0.01 unless configured); previously an omitted alpha
                silently reset self.alpha to 0.01.
        initial/delta are 4-vectors (x, y, z, t0) for both methods.
        Returns a 4-vector (x, y, z, t0).
        """
        if not isnan(alpha):
            self.cy_alpha = alpha

        min_coords = initial - delta
        max_coords = initial + delta

        self.read_traveltimes(
            min_coords=min_coords[:3],
            max_coords=max_coords[:3]
        )

        if method == "edt":
            return self._locate_edt(initial, min_coords, max_coords)

        bounds = np.stack([min_coords, max_coords]).T

        # RCP added some kwargs
        soln = scipy.optimize.differential_evolution(self.rms, bounds,
                                                     x0 = initial, # recent scipy allows an initial estimate 
                                                     strategy='best1bin', updating='immediate', 
                                                     maxiter=200, mutation=(0.3,1.0), recombination=0.7,
                                                     popsize=20, atol=0.01, tol=0.01, init='sobol',
                                                     seed=self.cy_locate_seed,
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


    def _locate_edt(self, initial, min_coords, max_coords):
        """
        EDT location following NonLinLoc's philosophy: rather than driving
        a local optimizer to a single minimum (which wanders on the flat /
        multi-modal EDT surface produced by weak or large-azimuthal-gap
        geometry), the hypocenter is taken as the MAXIMUM of the sampled
        EDT likelihood surface -- the mode of the PDF, exactly what NLL's
        oct-tree importance sampling reports.

        Implementation is a lightweight oct-tree analog: sample the search
        box, keep the best sample, contract the box around it, resample,
        and repeat. Because the estimate is the peak of a sampled surface
        (not an optimizer landing point), it is stable against the starting
        origin and does not require fabricating curvature (no edt_reg).
        A final local polish refines to sub-node precision. The companion
        sample_posterior() then reports the honest (often large) ellipsoid.

        Assumes traveltimes have already been read for the search volume.
        The sampling schedule (samples per stage, number of contraction
        stages, and shrink factor) is fixed internally; only locate_seed
        affects it, keeping results reproducible.
        """
        rng = np.random.default_rng(self.cy_locate_seed)

        lo = np.asarray(min_coords[:3], dtype=np.float64).copy()
        hi = np.asarray(max_coords[:3], dtype=np.float64).copy()

        # Fixed sampling schedule (oct-tree analog): samples per stage,
        # number of contraction stages, and box shrink factor per stage.
        n_per_stage = 1000
        n_stages = 7
        contract = 0.6

        best_x = 0.5 * (lo + hi)
        best_f = self.edt(best_x)      # edt() returns NEGATIVE log-L (misfit)

        for stage in range(n_stages):
            samples = rng.uniform(lo, hi, size=(n_per_stage, 3))
            # evaluate misfit at every sample; lower is better
            fvals = np.empty(n_per_stage, dtype=np.float64)
            for i in range(n_per_stage):
                fvals[i] = self.edt(samples[i])
            k = int(np.argmin(fvals))
            if fvals[k] < best_f:
                best_f = fvals[k]
                best_x = samples[k].copy()
            # contract the box around the current best for the next stage
            half = 0.5 * (hi - lo) * contract
            lo = np.maximum(best_x - half, np.asarray(min_coords[:3]))
            hi = np.minimum(best_x + half, np.asarray(max_coords[:3]))

        # local polish to sub-node precision around the sampled mode
        polished = scipy.optimize.minimize(
            self.edt, best_x,
            method='Nelder-Mead',
            options={'xatol': 0.05, 'fatol': 1e-4, 'maxiter': 200},
        )
        hypo = polished.x if polished.fun < best_f else best_x
        hypo = np.asarray(hypo, dtype=np.float64)

        t0 = self.origin_time(hypo)

        return np.append(hypo, t0)


    def residuals(self, hypocenter):
        """
        Per-arrival residuals t_obs - t0 - traveltime at a
        4-vector (x, y, z, t0) solution. Returns {key: residual}; arrivals
        whose traveltime grid does not cover the hypocenter are omitted.
        """
        hypocenter = np.asarray(hypocenter, dtype=np.float64)
        if self.cy_keys is None:
            self._prepare_workspace()
        self._fill_traveltimes(hypocenter[:3])

        residuals = {}
        tt_work = np.asarray(self.cy_tt_work)
        obs = np.asarray(self.cy_obs)
        for idx, key in enumerate(self.cy_keys):
            if np.isinf(tt_work[idx]):
                continue
            residuals[key] = obs[idx] - hypocenter[3] - tt_work[idx]

        return residuals


    def _edt_proposal_cov(self, mode, scale, delta):
        """
        Local covariance for the posterior proposal: -inv(H) where H is the
        finite-difference Hessian of the (sharpened) EDT log-likelihood at
        `mode`. Eigenvalues are floored and capped so that flat or
        non-concave directions -- routine on one-sided geometry, where the
        EDT surface is deliberately flat-topped -- give a broad but finite
        proposal rather than a singular one.
        """
        cdef int i, j

        step = np.maximum(0.002 * np.asarray(delta, dtype=np.float64), 0.05)
        H = np.zeros((3, 3), dtype=np.float64)

        def ll(x):
            return scale * self.edt_log_likelihood(
                np.ascontiguousarray(x, dtype=np.float64)
            )

        f0 = ll(mode)
        if not np.isfinite(f0):
            return None

        for i in range(3):
            for j in range(i, 3):
                if i == j:
                    p = mode.copy(); p[i] += step[i]
                    m = mode.copy(); m[i] -= step[i]
                    H[i, i] = (ll(p) - 2.0 * f0 + ll(m)) / (step[i] ** 2)
                else:
                    v = []
                    for si in (1.0, -1.0):
                        for sj in (1.0, -1.0):
                            p = mode.copy()
                            p[i] += si * step[i]
                            p[j] += sj * step[j]
                            v.append(ll(p))
                    H[i, j] = H[j, i] = (
                        (v[0] - v[1] - v[2] + v[3])
                        / (4.0 * step[i] * step[j])
                    )

        if not np.all(np.isfinite(H)):
            return None

        evals, evecs = np.linalg.eigh(-H)
        big = (np.max(delta) / 2.0) ** 2
        small = (0.002 * np.max(delta)) ** 2
        var = np.where(evals > 1e-8, 1.0 / np.maximum(evals, 1e-8), big)
        var = np.clip(var, small, big)
        return evecs @ np.diag(var) @ evecs.T


    def sample_posterior(self, hypocenter, delta, nsamples=4096, nscatter=1024,
                         seed=None, exponent=None, proposal="hessian",
                         inflate=3.0, rounds=2):
        """
        NLL-style posterior characterization around a solution: importance
        sampling of the EDT likelihood over hypocenter +/- delta (3-vectors
        or the first 3 elements of 4-vectors).

        Returns a dict with:
          scatter    : (nscatter, 3) posterior sample cloud (NLL SCAT analog)
          mean       : (3,) posterior expected hypocenter
          covariance : (3, 3) posterior covariance
          ellipsoid  : dict with 'semi_axes' (1-sigma lengths, sorted
                       descending) and 'axes' (unit vectors, rows matching
                       semi_axes) from the covariance eigendecomposition
          ess        : effective sample size of the importance weights
          proposal   : which proposal was actually used

        The raw EDT stack is a broad, heavy-tailed surface; following NLL's
        EDT^N sharpening, the posterior is taken proportional to
        stack^exponent. exponent=None (default) uses the number of arrivals,
        approximating the information content of N independent picks;
        pass exponent=1 for the raw (most conservative) EDT surface.

        proposal="hessian" (default) draws from a heavy-tailed multivariate
        t centred on the solution and scaled by the local curvature, then
        divides the proposal density out of the importance weights. This
        exists because the old uniform proposal degenerates badly whenever
        the sharpened posterior is narrow relative to delta: on
        well-constrained events the effective sample size collapses to ~1
        of nsamples, and the reported ellipsoid is then built from a single
        distinct point (semi-axes of exactly zero). Note that simply
        shrinking the uniform box is NOT a safe fix -- it truncates the
        posterior and understates uncertainty on weak geometry, which is
        the dangerous direction to be wrong in.

        proposal="uniform" restores the previous behaviour exactly.
        """
        rng = np.random.default_rng(seed)
        hypocenter = np.asarray(hypocenter, dtype=np.float64)[:3]
        delta = np.asarray(delta, dtype=np.float64)[:3]

        if exponent is None:
            exponent = max(len(self.cy_arrivals), 1)
        scale = exponent / self.cy_edt_exponent

        lo = hypocenter - delta
        hi = hypocenter + delta

        samples = None
        used = proposal

        if proposal == "hessian":
            cov = self._edt_proposal_cov(hypocenter, scale, delta)
            if cov is not None:
                cov = cov * (inflate ** 2)
                mu = hypocenter.copy()
                per = max(int(nsamples) // max(int(rounds), 1), 32)
                df = 4.0
                for r in range(max(int(rounds), 1)):
                    try:
                        L = np.linalg.cholesky(cov)
                    except np.linalg.LinAlgError:
                        samples = None
                        break
                    z = rng.standard_normal((per, 3))
                    g = rng.chisquare(df, per) / df
                    s = mu + (z / np.sqrt(g)[:, None]) @ L.T
                    s = s[np.all((s >= lo) & (s <= hi), axis=1)]
                    if len(s) < 16:
                        samples = None
                        break
                    logl = np.array([
                        scale * self.edt_log_likelihood(s_i) for s_i in s
                    ])
                    d = s - mu
                    quad = np.sum(d * np.linalg.solve(cov, d.T).T, axis=1)
                    logq = -0.5 * (df + 3.0) * np.log1p(quad / df)
                    logw = logl - logq
                    finite = np.isfinite(logw)
                    if not finite.any():
                        samples = None
                        break
                    samples, logw = s[finite], logw[finite]
                    if r < max(int(rounds), 1) - 1:
                        # adapt the proposal to the weighted moments
                        wa = np.exp(logw - logw.max())
                        wa /= wa.sum()
                        mu = wa @ samples
                        dev = samples - mu
                        c = (wa[:, None] * dev).T @ dev
                        ev, evec = np.linalg.eigh(c)
                        ev = np.clip(
                            ev, (0.002 * np.max(delta)) ** 2, np.max(delta) ** 2
                        )
                        cov = (evec @ np.diag(ev) @ evec.T) * (inflate ** 2)
            if samples is None:
                used = "uniform (hessian fallback)"

        if samples is None:
            # Uniform proposal: the density is constant, so it cancels and
            # the weights are the likelihood alone.
            samples = rng.uniform(lo, hi, size=(int(nsamples), 3))
            logw = np.array([
                scale * self.edt_log_likelihood(s_i) for s_i in samples
            ])
            finite = np.isfinite(logw)
            samples, logw = samples[finite], logw[finite]

        if len(samples) == 0:
            raise RuntimeError("No posterior samples with finite likelihood.")

        w = np.exp(logw - logw.max())
        w /= w.sum()
        ess = 1.0 / np.sum(w ** 2)

        mean = w @ samples
        dev = samples - mean
        cov_post = (w[:, None] * dev).T @ dev

        evals, evecs = np.linalg.eigh(cov_post)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        semi_axes = np.sqrt(np.clip(evals, 0, None))

        scatter_idx = rng.choice(len(samples), size=int(nscatter), p=w)
        scatter = samples[scatter_idx]

        return {
            "scatter": scatter,
            "mean": mean,
            "covariance": cov_post,
            "ellipsoid": {"semi_axes": semi_axes, "axes": evecs.T},
            "ess": ess,
            "proposal": used,
        }


    def quality(self, hypocenter):
        """
        NLL-style solution quality metrics at a 4-vector (x, y, z, t0):

          nobs             : arrivals whose grids cover the hypocenter
          rms              : classic (unweighted) RMS of residuals
          azimuthal_gap    : largest azimuthal gap in degrees (requires
                             station coordinates via add_stations())
          min_station_dist : epicentral distance to nearest used station
                             (same units as the grid for cartesian; km for
                             spherical, using great-circle distance)

        Azimuths/distances are computed in the locator's coordinate system:
        cartesian uses atan2 in the horizontal plane; spherical treats
        coordinates as (r, theta [colatitude], phi [longitude]) radians.
        """
        hypocenter = np.asarray(hypocenter, dtype=np.float64)
        residuals = self.residuals(hypocenter)
        nobs = len(residuals)

        out = {
            "nobs": nobs,
            "rms": np.nan,
            "azimuthal_gap": np.nan,
            "min_station_dist": np.nan,
        }
        if nobs == 0:
            return out

        res = np.array(list(residuals.values()))
        out["rms"] = float(np.sqrt(np.mean(res ** 2)))

        # station-geometry metrics
        azimuths = []
        dists = []
        used_stations = {key[:2] for key in residuals}
        for sta in used_stations:
            if sta not in self.cy_stations:
                continue
            coords = self.cy_stations[sta]
            if self.cy_coord_sys == "cartesian":
                dx = coords[0] - hypocenter[0]
                dy = coords[1] - hypocenter[1]
                azimuths.append(np.degrees(np.arctan2(dy, dx)) % 360.0)
                dists.append(np.sqrt(dx * dx + dy * dy))
            else:
                # spherical: (r, theta=colatitude, phi=longitude), radians
                lat1 = np.pi / 2 - hypocenter[1]
                lat2 = np.pi / 2 - coords[1]
                dlon = coords[2] - hypocenter[2]
                # great-circle distance (haversine)
                a = (
                    np.sin((lat2 - lat1) / 2) ** 2
                    + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
                )
                gc = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
                dists.append(gc * coords[0])
                az = np.degrees(np.arctan2(
                    np.sin(dlon) * np.cos(lat2),
                    np.cos(lat1) * np.sin(lat2)
                    - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
                )) % 360.0
                azimuths.append(az)

        if dists:
            out["min_station_dist"] = float(np.min(dists))
        if len(azimuths) >= 2:
            az = np.sort(azimuths)
            gaps = np.diff(np.append(az, az[0] + 360.0))
            out["azimuthal_gap"] = float(np.max(gaps))
        elif len(azimuths) == 1:
            out["azimuthal_gap"] = 360.0

        return out


    def locate_detailed(self, initial, delta, alpha=None, method="edt",
                        nsamples=4096, nscatter=1024, seed=None):
        """
        Convenience wrapper: locate, then characterize the solution. Returns
        a dict with 'hypocenter' (x, y, z, t0), 'quality', and (for EDT)
        'posterior' from sample_posterior(). Traveltimes for the search
        volume are read by locate() and reused for the posterior sampling.
        alpha=None uses the locator's current self.alpha.
        """
        initial = np.asarray(initial, dtype=np.float64)
        delta = np.asarray(delta, dtype=np.float64)

        hypo = self.locate(
            initial, delta,
            np.nan if alpha is None else alpha,
            method
        )

        out = {"hypocenter": hypo, "quality": self.quality(hypo)}

        if method == "edt":
            out["posterior"] = self.sample_posterior(
                hypo[:3], delta[:3],
                nsamples=nsamples, nscatter=nscatter, seed=seed
            )

        return out


    cpdef constants.REAL_t log_likelihood(
        EQLocator self,
        constants.REAL_t[:] model
    ):
        cdef constants.REAL_t   t_pred, residual
        cdef constants.REAL_t   log_likelihood = 0.0
        cdef tuple              key

        for key in self.cy_arrivals:
            if key not in self.cy_traveltimes:
                continue
            t_pred = (
                model[3]
                + self.cy_traveltimes[key].value(model[:3])
            )
            residual = self.cy_arrivals[key] - t_pred
            log_likelihood = log_likelihood + self.cy_residual_rvs[key].logpdf(residual)
        return (log_likelihood)
