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
from libc.math cimport sqrt as math_sqrt

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
        self.cy_variance_floor      = 0.01  # a-posteriori data variance floor (s^2)
        self.cy_default_pick_error  = 0.02  # seconds
        # NaN = "use the number of arrivals", matching NonLinLoc's EDT pdf
        # (the pair sum raised to the power N). Set a number to override.
        # This was 1.0, which is a monotone transform of the same surface
        # and so gave the same mode -- but it is NOT equivalent once the
        # EDT_OT_WT term is added, because that term is a fixed-scale
        # log-probability penalty added after the power, and its relative
        # weight depends on the exponent being N.
        self.cy_edt_exponent        = NAN
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
        # EDT_OT_WT: penalise the pdf by the spread of the per-arrival
        # origin-time estimates. On by default -- this is NonLinLoc's
        # recommended LOCMETH, and it replaces the quadratic
        # regularization this class used to carry.
        self.cy_edt_ot_wt           = True
        # NLL's EDT_OT_WT_FLOOR: the OT penalty cannot suppress the pdf by
        # more than a factor of 1e-5, so a wildly inconsistent trial point
        # is heavily disfavoured but never assigned exactly zero
        # probability (which would punch holes in the search surface).
        self.cy_edt_ot_wt_floor     = log(0.00001)
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

        self.cy_edge_axes = None
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
            value = 1e-5 # silent correction in this instance
        self.cy_default_pick_error = value

    @property
    def variance_floor(self):
        return self.cy_variance_floor

    @variance_floor.setter
    def variance_floor(self, value):
        if value < 0:
            raise ValueError("variance_floor must be >= 0")
        self.cy_variance_floor = value

    @property
    def edt_exponent(self):
        """
        Exponent applied to the EDT stack (NLL's pdf sharpening).

        None (the default) means "the number of arrivals", which is what
        NonLinLoc uses: its EDT pdf is the pair sum raised to the power N.
        Set a number to override; 1 gives the raw, most conservative EDT
        surface.

        This is not a cosmetic choice now that EDT_OT_WT is active. The
        origin-time penalty is added after the power, so the exponent sets
        the balance between the two terms.
        """
        if isnan(self.cy_edt_exponent):
            return None
        return self.cy_edt_exponent

    @edt_exponent.setter
    def edt_exponent(self, value):
        if value is None:
            self.cy_edt_exponent = NAN
            return
        if value <= 0:
            raise ValueError("edt_exponent must be > 0, or None for auto")
        self.cy_edt_exponent = value

    @property
    def edt_ot_wt(self):
        """
        Enable NonLinLoc's EDT_OT_WT (default True): penalise the EDT pdf
        by the weighted variance of the per-arrival origin-time estimates,
        so that points which satisfy many differential times but imply
        mutually inconsistent origin times are suppressed. This is what
        makes EDT pdfs compact without sacrificing outlier robustness.

        Set False for the plain EDT pair sum.
        """
        return bool(self.cy_edt_ot_wt)

    @edt_ot_wt.setter
    def edt_ot_wt(self, value):
        self.cy_edt_ot_wt = bool(value)

    @property
    def edt_ot_wt_floor(self):
        """
        Lower bound on the EDT_OT_WT log-penalty (default log(1e-5), as in
        NLL). Bounds how far the origin-time term can suppress the pdf, so
        badly inconsistent regions stay improbable rather than becoming
        exactly zero and tearing holes in the search surface.
        """
        return self.cy_edt_ot_wt_floor

    @edt_ot_wt_floor.setter
    def edt_ot_wt_floor(self, value):
        if value >= 0:
            raise ValueError("edt_ot_wt_floor must be negative (a log)")
        self.cy_edt_ot_wt_floor = value

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
        self.cy_sigma = np.maximum(
            np.array(
                [self.cy_pick_errors.get(key, self.cy_default_pick_error) for key in keys],
                dtype=np.float64,
            ),
            1e-5,
        )
        self.cy_tt_work   = np.empty(n, dtype=np.float64)
        self.cy_ot_work   = np.empty(n, dtype=np.float64)

        return True


    cdef int _fill_traveltimes(EQLocator self, constants.REAL_t[:] hypo_xyz):
        """
        Interpolate every arrival traveltime at
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


    cdef constants.REAL_t _effective_exponent(EQLocator self, int n):
        """
        Exponent applied to the EDT stack. NaN (the default) means "use the
        number of arrivals", which is what NonLinLoc does: its EDT pdf is
        the pair sum raised to the power N, and its oct-tree search works
        on that. Any explicit value overrides it.

        The arrival COUNT is used rather than the count of arrivals whose
        grids cover the trial point, deliberately. Using the covered count
        makes the exponent a function of position, which puts a step in the
        likelihood at every grid edge -- the discontinuity NLL flags in its
        own source (20200619) as a reason not to weight by the reading
        count.
        """
        if isnan(self.cy_edt_exponent):
            return <constants.REAL_t> n
        return self.cy_edt_exponent


    cpdef constants.REAL_t edt_log_likelihood(
        EQLocator self,
        constants.REAL_t[:] hypo_xyz
    ):
        """
        NonLinLoc's Equal Differential Time log-likelihood at a trial
        hypocenter (3 spatial coordinates; origin time cancels).

        For every pair of arrivals (a, b), with
        s^2 = pick_error^2 + (alpha * tt)^2 combining pick uncertainty with
        a fractional traveltime (velocity-model) error:

            r_ab  = (t_obs_a - t_obs_b) - (tt_a - tt_b)
            v_ab  = s_a^2 + s_b^2
            p_ab  = exp(-0.5 * r_ab^2 / v_ab) / sqrt(v_ab)
            L     = sum_ab p_ab

        and the returned log-likelihood is

            N * log(L / n_pairs)  +  ot_var_weight

        NOTE THE FACTOR OF 0.5. Earlier versions used
        exp(-r^2 / v) with no 0.5, which is not the density of a difference
        of two Gaussians and made the kernel narrower than intended by a
        factor of sqrt(2) in effective sigma -- i.e. every pick was being
        treated as sqrt(2) times more precise than its stated error. NLL's
        reference implementation (NLLocLib.c, CalcSolutionQuality_EDT:
        `prob = exp(-0.5 * edt_misfit * edt_misfit * weight2)`) has the 0.5.
        Correcting it widens the likelihood, so pick errors and alpha tuned
        against the old kernel will now behave as if they were sqrt(2)
        larger; re-check them if you had them dialled in.

        ot_var_weight is the EDT_OT_WT term (LOCMETH EDT_OT_WT). Each
        arrival implies its own origin time, ot_i = t_obs_i - tt_i. At the
        true hypocenter these agree; at a point that happens to satisfy many
        differential times by coincidence they scatter. So NLL penalises
        the pdf by the spread of those origin-time estimates, weighted by
        each arrival's accumulated EDT probability:

            w_i           = sum_{j != i} p_ij
            ot_var        = weighted variance of ot_i under w_i
            ot_var_weight = -ot_var / mean(s_i^2),  floored at log(1e-5)

        This replaces the quadratic regularization term this class used to
        carry. That term added an L2 differential-time misfit, which
        reintroduces exactly the outlier sensitivity EDT exists to avoid --
        hence its failure on one-sided geometry. The OT-variance term
        instead sharpens the pdf using a statistic computed from the robust
        pair weights, so a single bad pick contributes negligible w_i and
        barely moves it.

        The term is added AFTER the power of N, not inside it, matching
        NLL: it is a fixed-scale log-probability penalty, not something
        that grows with the reading count.
        """
        cdef int    ia, ib, n, npairs = 0, nvalid = 0
        cdef constants.REAL_t tta, ttb, r, va, vb, vv, prob
        cdef constants.REAL_t stack = 0.0
        cdef constants.REAL_t ot_i, ot_w
        cdef constants.REAL_t ot_sum = 0.0, ot_2_sum = 0.0, ot_wsum = 0.0
        cdef constants.REAL_t sig2_sum = 0.0
        cdef constants.REAL_t ot_mean, ot_var, ot_var_weight = 0.0
        cdef constants.REAL_t alpha_sq = self.cy_alpha * self.cy_alpha

        if self.cy_keys is None:
            self._prepare_workspace()

        n = len(self.cy_keys)
        if self._fill_traveltimes(hypo_xyz) < 2:
            return -INFINITY

        if self.cy_edt_ot_wt:
            for ia in range(n):
                self.cy_ot_work[ia] = 0.0

        for ia in range(n - 1):
            tta = self.cy_tt_work[ia]
            if isinf(tta):
                continue
            va = self.cy_variance_floor + self.cy_sigma[ia]*self.cy_sigma[ia] + alpha_sq*tta*tta
            for ib in range(ia + 1, n):
                ttb = self.cy_tt_work[ib]
                if isinf(ttb):
                    continue
                vb = self.cy_variance_floor + self.cy_sigma[ib]*self.cy_sigma[ib] + alpha_sq*ttb*ttb
                vv = va + vb
                r = (self.cy_obs[ia] - self.cy_obs[ib]) - (tta - ttb)
                prob = exp(-0.5 * (r * r) / vv) / sqrt(vv)
                stack += prob
                npairs += 1
                if self.cy_edt_ot_wt:
                    # Accumulate each pair's probability onto BOTH arrivals.
                    # NLL's EDT_OT_WT credits it only to the lower-indexed
                    # one, so its last arrival contributes zero weight to
                    # the origin-time statistics and the result depends on
                    # arrival ordering. That looks like an oversight rather
                    # than intent: the EDT_OT_WT_ML branch of the same
                    # function accumulates onto both, under a comment
                    # reading "AJL 20070326 bug fix!". We follow the fixed
                    # branch.
                    self.cy_ot_work[ia] += prob
                    self.cy_ot_work[ib] += prob

        if npairs == 0 or stack <= 0.0:
            return -INFINITY

        # Normalise by the pair count of the FULL arrival set, not the pairs
        # that happen to be covered at this trial point. npairs is a function
        # of position: as arrivals fall outside their grids the mean over the
        # survivors rises, so the likelihood improves purely because the
        # badly-fitting readings stopped being counted. This is the same
        # discontinuity the exponent already avoids by using len(cy_keys)
        # (see _effective_exponent); the divisor has to match.
        # Can experiment with this on/off..
        npairs = (n * (n - 1)) // 2

        if self.cy_edt_ot_wt:
            for ia in range(n):
                tta = self.cy_tt_work[ia]
                if isinf(tta):
                    continue
                sig2_sum += self.cy_variance_floor + self.cy_sigma[ia]*self.cy_sigma[ia] + alpha_sq*tta*tta
                nvalid += 1
                ot_i = self.cy_obs[ia] - tta
                ot_w = self.cy_ot_work[ia]
                ot_sum += ot_w * ot_i
                ot_2_sum += ot_w * ot_i * ot_i
                ot_wsum += ot_w
            if ot_wsum > 0.0 and nvalid > 0:
                ot_mean = ot_sum / ot_wsum
                ot_var = ot_2_sum / ot_wsum - ot_mean * ot_mean
                if ot_var < 0.0:        # rounding on a perfectly consistent fit
                    ot_var = 0.0
                ot_var_weight = -ot_var / (sig2_sum / nvalid)
                if ot_var_weight < self.cy_edt_ot_wt_floor:
                    ot_var_weight = self.cy_edt_ot_wt_floor

        return (
            self._effective_exponent(n) * log(stack / npairs)
            + ot_var_weight
        )


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


    def edt_ot_stats(EQLocator self, hypo_xyz):
        """
        Origin-time statistics from the EDT pair weights at a fixed
        hypocenter. Returns (mean, variance, weights, ot) where `ot` holds
        each valid arrival's implied origin time t_obs - tt and `weights`
        holds its accumulated pair probability.

        This is the same quantity the EDT_OT_WT term inside
        edt_log_likelihood computes; it is exposed separately (and
        vectorised rather than looped, since it runs once per solution
        rather than once per likelihood evaluation) so the origin time and
        its scatter can be reported and inspected.
        """
        hypo_xyz = np.ascontiguousarray(hypo_xyz[:3], dtype=np.float64)
        if self.cy_keys is None:
            self._prepare_workspace()
        if self._fill_traveltimes(hypo_xyz) < 2:
            return np.nan, np.nan, np.zeros(0), np.zeros(0)

        alpha_sq = self.cy_alpha * self.cy_alpha
        tt_all = np.asarray(self.cy_tt_work)
        valid = np.isfinite(tt_all)
        tt = tt_all[valid]
        obs = np.asarray(self.cy_obs)[valid]
        sig = np.asarray(self.cy_sigma)[valid]

        sig2 = sig * sig + alpha_sq * tt * tt
        vv = sig2[:, None] + sig2[None, :]
        r = (obs[:, None] - obs[None, :]) - (tt[:, None] - tt[None, :])
        prob = np.exp(-0.5 * (r * r) / vv) / np.sqrt(vv)
        np.fill_diagonal(prob, 0.0)

        w = prob.sum(axis=1)
        ot = obs - tt
        wsum = w.sum()
        if wsum <= 0:
            return np.nan, np.nan, w, ot
        mean = float(w @ ot / wsum)
        var = float(w @ (ot - mean) ** 2 / wsum)
        return mean, var, w, ot


    cpdef constants.REAL_t origin_time(EQLocator self, constants.REAL_t[:] hypo_xyz):
        """
        Origin time at a fixed hypocenter.

        With edt_ot_wt on (the default) this is NonLinLoc's EDT_OT_WT
        estimator: the mean of the per-arrival origin times t_obs - tt,
        weighted by each arrival's accumulated EDT pair probability
        (NLLocLib.c: `*potime = ot_sum / ot_weight`). It is robust for the
        same reason the EDT stack is -- an outlier pick satisfies few pairs,
        so its weight collapses -- and it is the estimator consistent with
        the objective actually being maximised.

        With edt_ot_wt off, falls back to the previous behaviour: the
        inverse-variance weighted MEDIAN. That is also robust, but it is a
        step function of position, which makes the recovered origin time
        jump between picks as the hypocenter moves and would give a lumpy
        marginal if t0 were ever added to the posterior.
        """
        cdef int idx, n

        if self.cy_keys is None:
            self._prepare_workspace()

        n = len(self.cy_keys)
        if self._fill_traveltimes(hypo_xyz) == 0:
            return np.nan

        if self.cy_edt_ot_wt:
            mean, _, w, _ = self.edt_ot_stats(hypo_xyz)
            if np.isfinite(mean):
                return mean
            # fall through to the median if the pair weights degenerate
            # (fewer than two usable arrivals)

        alpha_sq = self.cy_alpha * self.cy_alpha
        tt = np.asarray(self.cy_tt_work)
        valid = np.isfinite(tt)
        t0s = np.asarray(self.cy_obs)[valid] - tt[valid]
        sig = np.asarray(self.cy_sigma)[valid]
        w = 1.0 / (sig * sig + alpha_sq * tt[valid] * tt[valid])

        order = np.argsort(t0s)
        t0s, w = t0s[order], w[order]
        cw = np.cumsum(w)
        return float(t0s[np.searchsorted(cw, 0.5 * cw[cw.shape[0] - 1])])

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
        str method="l1",
        np.ndarray bounds=None
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

        if bounds is None:
            min_coords = initial - delta
            max_coords = initial + delta
        else:
            # Explicit, possibly ASYMMETRIC search interval: bounds[0] = lo,
            # bounds[1] = hi, each a 4-vector (x, y, z, t0). Required wherever
            # a physical limit truncates one side of the box -- writing that
            # as centre +/- half-width reproduces the interval but silently
            # moves the centre off the anchor.
            bounds = np.asarray(bounds, dtype=np.float64)
            min_coords = bounds[0].copy()
            max_coords = bounds[1].copy()

        self.read_traveltimes(
            min_coords=min_coords[:3],
            max_coords=max_coords[:3]
        )

        if method == "edt":
            return self._locate_edt(initial, min_coords, max_coords)

        # the below is for the L1 solver only
        bounds = np.stack([min_coords, max_coords]).T
        soln = scipy.optimize.differential_evolution(self.rms, bounds,
                                                     x0 = initial,
                                                     strategy='best1bin', updating='immediate',
                                                     maxiter=200, mutation=(0.3,1.0), recombination=0.7,
                                                     popsize=20, atol=0.01, tol=0.01, init='sobol',
                                                     seed=self.cy_locate_seed,
                                                     polish=False)

        # Polish (find the bottom of the basin)
        polished = scipy.optimize.minimize(
            self.rms, soln.x,
            method='Nelder-Mead',
            bounds=bounds, # added bounds
            options={
                'xatol': 0.05,    # 50 m / 50 ms — tighter than DE could give
                'fatol': 0.005,   # 5 ms RMS resolution
                'maxiter': 100,
            },
        )

        final_x = polished.x if polished.fun < soln.fun else soln.x

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
        origin and does not require fabricating curvature.
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

        # Seed the incumbent at the ANCHOR, not the box centre.
        best_x = np.clip(np.asarray(initial[:3], dtype=np.float64), lo, hi)
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
            bounds=scipy.optimize.Bounds(
                np.asarray(min_coords[:3], dtype=np.float64),
                np.asarray(max_coords[:3], dtype=np.float64)),            
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


    def _metric_scale(self, x):
        """
        Diagonal factors converting a small offset in GRID coordinates at
        position `x` into physical kilometres.

        Cartesian grids are already in km, so this is (1, 1, 1). Spherical
        grids use (r [km], theta [rad, colatitude], phi [rad, longitude]),
        where an offset (dr, dtheta, dphi) spans
        (dr, r*dtheta, r*sin(theta)*dphi) kilometres.

        ALL curvature, covariance and ellipsoid computation below is done
        in this local km frame. Mixing km with radians -- as earlier
        versions did -- makes finite-difference steps, eigenvalue floors
        and ellipsoid semi-axes meaningless on spherical grids: floors
        derived from a radial half-width in km were applied to angular
        variances in rad^2, inflating horizontal sigma to hundreds of km,
        which in turn made almost every proposal draw fall outside the
        search box and silently forced the uniform fallback. The reported
        confidence ellipsoid was likewise an eigendecomposition of a
        matrix whose entries had three different units.

        The km frame's axes are r_hat, theta_hat, phi_hat (up, south,
        east): mutually orthogonal, so eigenvectors computed in it are
        genuine physical principal axes.
        """
        if self.cy_coord_sys == "cartesian":
            return np.ones(3, dtype=np.float64)
        r = float(x[0])
        # guard the poles, where the phi metric degenerates
        sin_theta = max(abs(np.sin(float(x[1]))), 1e-6)
        return np.array([1.0, r, r * sin_theta], dtype=np.float64)


    def _edt_proposal_cov(self, mode, scale, delta):
        """
        Local covariance IN KM^2 for the posterior proposal: -inv(H), where
        H is the finite-difference Hessian of the (sharpened) EDT
        log-likelihood at `mode` taken with respect to the local km frame
        (see _metric_scale). Eigenvalues are floored and capped so that
        flat or non-concave directions -- routine on one-sided geometry,
        where the EDT surface is deliberately flat-topped -- give a broad
        but finite proposal rather than a singular one.

        Returns a 3x3 covariance in km^2 in the (r_hat, theta_hat, phi_hat)
        frame (plain xyz for cartesian grids), or None if the surface is
        unusable at `mode`.
        """
        cdef int i, j

        s = self._metric_scale(mode)
        delta_km = np.abs(np.asarray(delta, dtype=np.float64)[:3]) * s

        # Finite-difference step: 0.2% of the search half-width, floored at
        # 50 m so the stencil is not lost in interpolation noise, and capped
        # at a quarter of the box so it stays local and inside the grids.
        step_km = np.clip(
            0.002 * delta_km,
            0.05,
            np.maximum(0.25 * delta_km, 0.05)
        )
        step = step_km / s      # the same displacement, in grid units

        H = np.zeros((3, 3), dtype=np.float64)

        def ll(x):
            return scale * self.edt_log_likelihood(
                np.ascontiguousarray(x, dtype=np.float64)
            )

        f0 = ll(mode)
        if not np.isfinite(f0):
            return None

        def at(offsets):
            """log-likelihood at mode + offsets (grid units), or None when
            the point falls outside the traveltime grids."""
            q = mode.copy()
            for axis, k in offsets:
                q[axis] += k * step[axis]
            val = ll(q)
            return val if np.isfinite(val) else None

        # Axes on which the solution sits against the edge of the region
        # for which traveltimes were read. This is routine, not
        # exceptional: when an axis is unresolved -- depth, on a network
        # with no near-source station -- the likelihood is monotone along
        # it, so the mode runs to the boundary of the search volume and
        # stops there. A centred stencil then steps off the grid.
        #
        # Previously ANY such step made the Hessian non-finite and this
        # routine returned None, dropping the sampler into the uniform
        # fallback. That is what produced fallback rates of 50-80%, and it
        # happened precisely on the geometries where a good proposal
        # matters most. The stencil now goes one-sided on the offending
        # axis instead of discarding all three.
        edge = np.zeros(3, dtype=bool)

        for i in range(3):
            plus = at([(i, 1.0)])
            minus = at([(i, -1.0)])
            if plus is not None and minus is not None:
                H[i, i] = (plus - 2.0 * f0 + minus) / (step_km[i] ** 2)
            else:
                # One-sided second difference on whichever side is inside
                # the grid: f(x) - 2f(x+h) + f(x+2h).
                edge[i] = True
                sgn = 1.0 if plus is not None else -1.0
                f1 = plus if plus is not None else minus
                f2 = at([(i, 2.0 * sgn)])
                if f1 is None or f2 is None:
                    # Both directions blocked: no curvature information on
                    # this axis. Zero leaves it to the eigenvalue floor
                    # below, which gives a broad, box-limited proposal --
                    # the honest answer for an axis the data cannot see.
                    H[i, i] = 0.0
                else:
                    H[i, i] = (f0 - 2.0 * f1 + f2) / (step_km[i] ** 2)

        for i in range(3):
            for j in range(i + 1, 3):
                v = [
                    at([(i, si), (j, sj)])
                    for si in (1.0, -1.0) for sj in (1.0, -1.0)
                ]
                if any(x is None for x in v):
                    # A blocked corner means no reliable mixed derivative.
                    # Zero assumes the axes are locally uncorrelated, which
                    # costs proposal efficiency but keeps the matrix
                    # usable; failing loses everything.
                    H[i, j] = H[j, i] = 0.0
                else:
                    H[i, j] = H[j, i] = (
                        (v[0] - v[1] - v[2] + v[3])
                        / (4.0 * step_km[i] * step_km[j])
                    )

        self.cy_edge_axes = edge

        if not np.all(np.isfinite(H)):
            return None

        evals, evecs = np.linalg.eigh(-H)

        # Bound each eigen-direction by the extent of the search box ALONG
        # THAT DIRECTION, not by the largest half-width of the box. A flat
        # (non-concave) direction gets the fallback variance `big`, and
        # with a single scalar `big = max(delta_km)/2` a flat DEPTH
        # direction in a box 15 km deep and 30 km wide was assigned a 15 km
        # sigma -- wider than the box it has to be sampled inside. Most
        # draws then landed outside and the whole proposal bailed out to
        # the uniform fallback, which is what drove the 50-80% fallback
        # rates on depth-unresolved geometry.
        half = 0.5 * np.sqrt(((evecs.T * delta_km) ** 2).sum(axis=1))
        big = half ** 2
        small = np.maximum((0.002 * half) ** 2, 1.0e-4)        # >= 10 m
        var = np.where(evals > 1e-8, 1.0 / np.maximum(evals, 1e-8), big)
        var = np.clip(var, small, big)
        return evecs @ np.diag(var) @ evecs.T


    def sample_posterior(self, hypocenter, delta, nsamples=4096, nscatter=1024,
                         seed=None, exponent=None, proposal="hessian",
                         inflate=3.0, rounds=2, center="solution",
                         search_delta=None, include_time=True):
        """
        NLL-style posterior characterization around a solution: importance
        sampling of the EDT likelihood over hypocenter +/- delta (3-vectors
        or the first 3 elements of 4-vectors).

        Returns a dict with:
          scatter       : (nscatter, 3) posterior sample cloud in GRID
                          coordinates (NLL SCAT analog)
          mean          : (3,) posterior expected hypocenter, grid coords
                          (NLL's "expectation" solution)
          center        : which point the reported covariance is taken
                          about, "solution" (default) or "mean"
          mode_minus_mean_km : (3,) offset between the two, in km. Large
                          values mean a strongly skewed posterior, and are
                          the signal that the choice of `center` matters
          covariance    : (3, 3) posterior covariance in RAW GRID units.
                          On a spherical grid these units are mixed
                          (km, rad, rad) -- retained for backward
                          compatibility only. Prefer covariance_km.
          covariance_km : (3, 3) posterior covariance in km^2, expressed in
                          the local orthonormal frame at the solution
                          (r_hat, theta_hat, phi_hat = up, south, east for
                          spherical grids; xyz for cartesian).
          ellipsoid     : legacy raw-coordinate eigendecomposition of
                          `covariance` (deprecated; meaningless units on
                          spherical grids)
          ellipsoid_km  : dict with 'semi_axes' (1-sigma lengths in km,
                          sorted descending), 'axes' (unit vectors in the
                          local km frame, rows matching semi_axes) and
                          'volume_km3'. THIS is the physically meaningful
                          error ellipsoid; scale the semi-axes by the
                          appropriate chi-square quantile (1.878 for a 68%
                          three-dimensional confidence ellipsoid) before
                          reporting a confidence level.
          metric_scale  : (3,) the grid -> km factors used
          box_limited   : True when the posterior is wide enough relative
                          to the search box that the box, not the data, is
                          setting the reported width. The uncertainty is
                          then a LOWER BOUND and should be reported as
                          unresolved rather than as a number.
          box_fill      : per-axis sigma / search half-width, measured
                          against `search_delta` if given, else `delta`
          at_search_edge: per-axis flag, True where the solution sits
                          against the edge of the searched volume. Means
                          the likelihood is monotone along that axis: the
                          data do not locate it, and the reported value is
                          a boundary, not a minimum.
          time          : origin time at the reported solution (seconds, in
                          the same reference as the arrival times)
          time_sigma    : 1-sigma uncertainty on the origin time
          covariance4_km_s : (4, 4) covariance over
                          (axis1, axis2, axis3 [km], t0 [s]) about the same
                          point as `covariance_km`. The EDT likelihood is
                          origin-time independent, so adding t0 does not
                          change the spatial block -- it adds the origin
                          time uncertainty and, more usefully, its
                          covariance with depth.
          depth_time_correlation : correlation between the vertical axis
                          and origin time. Near -1 or +1 means depth and t0
                          are trading off and neither is independently
                          determined, which is the normal state of affairs
                          without a near-source station.
          sigma_rel_mcse: Monte Carlo standard error on the reported sigma,
                          as a fraction of sigma (~1/sqrt(2*ESS))
          ess           : effective sample size of the importance weights
          proposal      : which proposal was actually used

        The raw EDT stack is a broad, heavy-tailed surface; following NLL's
        EDT^N sharpening, the posterior is taken proportional to
        stack^exponent. exponent=None (default) uses the number of arrivals,
        matching NLL's formulation; pass exponent=1 for the raw (most
        conservative) EDT surface.

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

        rounds = 2 is the safe call. Above this can be counter-productive.

        proposal="uniform" restores the original behaviour.
        """
        rng = np.random.default_rng(seed)
        hypocenter = np.asarray(hypocenter, dtype=np.float64)[:3]
        delta = np.asarray(delta, dtype=np.float64)[:3]

        # a-posteriori data variance floor: the reported uncertainty must reflect
        # how badly the data actually fit, not only the a-priori pick/alpha terms.
        # Without this a user setting pick_error~0 and alpha=0 gets a spuriously
        # sharp posterior even with large residuals. Measure the residual variance
        # at the solution and floor every pair's variance with it (NLL-consistent:
        # the observational covariance carries the real misfit scale).
        _saved_vfloor = self.cy_variance_floor
        if self.cy_keys is None:
            self._prepare_workspace()
        if self._fill_traveltimes(hypocenter[:3]) >= 2:
            tt = np.asarray(self.cy_tt_work)
            obs = np.asarray(self.cy_obs)
            finite = ~np.isinf(tt)
            if finite.sum() > 4:
                r = obs[finite] - tt[finite]
                r = r - np.median(r)          # remove origin-time (robust center)
                dof = finite.sum() - 4
                self.cy_variance_floor = float(np.sum(r*r) / dof)

        if exponent is None:
            exponent = max(len(self.cy_arrivals), 1)
        # edt_log_likelihood already applies its own exponent (the arrival
        # count by default), so rescale rather than re-apply. With both at
        # the default this is exactly 1 and the likelihood passes through
        # untouched. NOTE: the EDT_OT_WT term is scaled along with the
        # stack here, whereas NLL adds it outside the power. They coincide
        # at the default; a non-default `exponent` shifts the balance
        # slightly in favour of the origin-time penalty.
        _eff = self.edt_exponent
        if _eff is None:
            _eff = max(len(self.cy_keys or self.cy_arrivals), 1)
        scale = exponent / _eff

        # grid -> km factors, evaluated once at the solution. The proposal
        # region is small enough that treating them as constant over it is
        # accurate, and the resulting constant Jacobian cancels out of the
        # self-normalised importance weights.
        s = self._metric_scale(hypocenter)
        delta_km = np.abs(delta) * s

        lo = hypocenter - delta
        hi = hypocenter + delta

        samples = None
        used = proposal

        if proposal == "hessian":
            cov = self._edt_proposal_cov(hypocenter, scale, delta)   # km^2
            if cov is not None:
                cov = cov * (inflate ** 2)
                # Cap the inflated proposal at half the search box. Draws
                # outside the box are rejected, so an inflated proposal that
                # is wide relative to a contracted box loses almost every
                # sample and trips the < 16 survivors bail-out into the
                # uniform fallback which is exactly the degenerate path
                # the hessian proposal exists to avoid.
                _sig = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
                _cap = np.min(np.maximum(0.5 * delta_km, 1e-6) / _sig)
                if _cap < 1.0:
                    cov = cov * (_cap ** 2)
                mu = hypocenter.copy()             # grid coordinates
                per = max(int(nsamples) // max(int(rounds), 1), 32)
                df = 4.0
                var_lo = max((0.002 * np.min(delta_km)) ** 2, 1.0e-4)
                var_hi = np.max(delta_km) ** 2
                for r in range(max(int(rounds), 1)):
                    try:
                        L = np.linalg.cholesky(cov)
                    except np.linalg.LinAlgError:
                        samples = None
                        break
                    z = rng.standard_normal((per, 3))
                    g = rng.chisquare(df, per) / df
                    y = (z / np.sqrt(g)[:, None]) @ L.T     # km offsets from mu
                    cand = mu + y / s                       # back to grid units
                    keep = np.all((cand >= lo) & (cand <= hi), axis=1)
                    cand, y = cand[keep], y[keep]
                    if len(cand) < 16:
                        samples = None
                        break
                    logl = np.array([
                        scale * self.edt_log_likelihood(c_i) for c_i in cand
                    ])
                    quad = np.sum(y * np.linalg.solve(cov, y.T).T, axis=1)
                    logq = -0.5 * (df + 3.0) * np.log1p(quad / df)
                    logw = logl - logq
                    finite = np.isfinite(logw)
                    if not finite.any():
                        samples = None
                        break
                    samples, y, logw = cand[finite], y[finite], logw[finite]
                    if r < max(int(rounds), 1) - 1:
                        # adapt the proposal to the weighted moments (in km)
                        wa = np.exp(logw - logw.max())
                        wa /= wa.sum()
                        mu_y = wa @ y
                        mu = mu + mu_y / s
                        dev = y - mu_y
                        c = (wa[:, None] * dev).T @ dev
                        ev, evec = np.linalg.eigh(c)
                        ev = np.clip(ev, var_lo, var_hi)
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
        cov_mean = (w[:, None] * dev).T @ dev

        # Second moment about the SOLUTION rather than about the posterior
        # mean. These differ by the outer product of the mode-mean offset,
        # and on skewed EDT posteriors that offset is not small: on
        # one-sided geometry it routinely equals the semi-major axis
        # itself. Reporting a mean-centred covariance alongside a
        # mode-centred hypocenter gives an ellipsoid that is not centred on
        # the location it is attached to, and coverage collapses
        # accordingly. Whichever point is published, the moment must be
        # taken about that point.
        dev0 = samples - np.asarray(hypocenter, dtype=np.float64)
        cov_solution = (w[:, None] * dev0).T @ dev0

        cov_post = cov_solution if center == "solution" else cov_mean

        # Physical covariance and ellipsoid in the local orthonormal km
        # frame. For a cartesian grid s == (1, 1, 1) and these are
        # identical to the raw-coordinate versions.
        dev_km = (dev0 if center == "solution" else dev) * s
        cov_km = (w[:, None] * dev_km).T @ dev_km
        dev_mean_km = dev * s
        cov_mean_km = (w[:, None] * dev_mean_km).T @ dev_mean_km
        evals_km, evecs_km = np.linalg.eigh(cov_km)
        order = np.argsort(evals_km)[::-1]
        evals_km, evecs_km = evals_km[order], evecs_km[:, order]
        semi_km = np.sqrt(np.clip(evals_km, 0, None))

        evals, evecs = np.linalg.eigh(cov_post)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        semi_axes = np.sqrt(np.clip(evals, 0, None))

        # Is the posterior actually bounded by the data, or by the search
        # box? On geometry with no depth resolution the EDT surface is flat
        # across the entire search volume, and then the reported sigma
        # measures the box we chose rather than anything the arrivals
        # constrain -- widen the box and sigma grows with it. This is not
        # something a sampler can fix, and inflating the ellipsoid would
        # only disguise it, so it is surfaced as a flag instead. NLL has
        # the same property: its pdf is likewise truncated by LOCGRID.
        # Measured against the ORIGINAL search volume, not against `delta`.
        # Callers routinely contract the proposal to a few sigma before the
        # final pass, and comparing sigma with a box that was constructed
        # to be ~4 sigma wide flags every event as box-limited. Pass
        # search_delta=<the original half-widths> to get a meaningful
        # answer; it defaults to delta for callers that do not contract.
        _sd = delta if search_delta is None else np.abs(
            np.asarray(search_delta, dtype=np.float64)[:3]
        )
        box_fill = np.sqrt(np.clip(np.diag(cov_km), 0, None)) / np.maximum(
            np.abs(_sd) * s, 1e-9
        )
        box_limited = bool(np.any(box_fill > 0.25))

        # ---- origin time: promote the posterior to four dimensions -------
        #
        # The EDT likelihood is origin-time independent by construction, so
        # the SPATIAL posterior above already marginalises over t0 and none
        # of the numbers change by adding it. What t0 buys is the two
        # things that were previously simply not reported: an uncertainty
        # on the origin time itself, and its covariance with depth. On a
        # network without a near-source station those two trade off almost
        # completely -- a deeper origin with an earlier t0 fits the same
        # arrivals -- and the correlation coefficient is the honest way to
        # say so. It also lets a downstream consumer propagate the origin
        # time properly instead of assuming it is exact.
        #
        # t0 at each sample is the same estimator origin_time() uses, so
        # the marginal is consistent with the reported hypocenter.
        time_solution = time_sigma = np.nan
        cov4 = None
        depth_time_corr = np.nan
        if include_time:
            t0s = np.empty(len(samples), dtype=np.float64)
            for _i in range(len(samples)):
                t0s[_i] = self.origin_time(
                    np.ascontiguousarray(samples[_i], dtype=np.float64)
                )
            _finite = np.isfinite(t0s)
            if _finite.sum() >= 4:
                t0_solution = self.origin_time(
                    np.ascontiguousarray(hypocenter, dtype=np.float64)
                )
                _w4 = w[_finite] / w[_finite].sum()
                _s4 = samples[_finite]
                _t4 = t0s[_finite]
                if center == "solution" and np.isfinite(t0_solution):
                    _c3 = np.asarray(hypocenter, dtype=np.float64)
                    _ct = t0_solution
                else:
                    _c3 = _w4 @ _s4
                    _ct = float(_w4 @ _t4)
                _d4 = np.column_stack([(_s4 - _c3) * s, _t4 - _ct])
                cov4 = (_w4[:, None] * _d4).T @ _d4
                time_solution = (
                    t0_solution if np.isfinite(t0_solution)
                    else float(_w4 @ _t4)
                )
                time_sigma = float(np.sqrt(max(cov4[3, 3], 0.0)))
                # correlation between the vertical axis and origin time.
                # Axis 0 is r (up) for spherical grids, axis 2 is z for
                # cartesian; both are the vertical one.
                _iz = 0 if self.cy_coord_sys != "cartesian" else 2
                _den = math_sqrt(max(cov4[_iz, _iz], 0.0)) * time_sigma
                if _den > 0:
                    depth_time_corr = float(cov4[_iz, 3] / _den)

        # Monte Carlo standard error on the reported sigma, as a FRACTION
        # of sigma. Importance-sampling variance estimates have relative
        # error ~1/sqrt(2*ESS), so at ESS=80 the sigma is only good to ~8%
        # as real.
        sigma_rel_mcse = float(1.0 / np.sqrt(2.0 * max(ess, 1.0)))

        if not np.all(np.isfinite(w)) or w.sum() <= 0:
            raise RuntimeError("posterior weights degenerate (check pick errors / alpha)")
        w = w / w.sum()

        scatter_idx = rng.choice(len(samples), size=int(nscatter), p=w)
        scatter = samples[scatter_idx]

        # Restore da floor
        self.cy_variance_floor = _saved_vfloor

        return {
            "scatter": scatter,
            "mean": mean,
            "covariance": cov_post,
            "covariance_km": cov_km,
            "covariance_km_about_mean": cov_mean_km,
            "center": center,
            "mode_minus_mean_km": (
                np.asarray(hypocenter, dtype=np.float64) - mean
            ) * s,
            "ellipsoid": {"semi_axes": semi_axes, "axes": evecs.T},
            "ellipsoid_km": {
                "semi_axes": semi_km,
                "axes": evecs_km.T,
                "volume_km3": float(4.0 / 3.0 * np.pi * np.prod(semi_km)),
            },
            "metric_scale": s,
            "ess": ess,
            "proposal": used,
            "time": time_solution,
            "time_sigma": time_sigma,
            "covariance4_km_s": cov4,
            "depth_time_correlation": depth_time_corr,
            "sigma_rel_mcse": sigma_rel_mcse,
            "box_limited": box_limited,
            "box_fill": box_fill,
            "at_search_edge": np.asarray(
                self.cy_edge_axes if self.cy_edge_axes is not None
                else np.zeros(3, dtype=bool)
            ),
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
