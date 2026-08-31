cimport numpy as np

from . cimport constants
from . cimport fields


cdef class EQLocator(object):
    cdef str                     cy_coord_sys
    cdef dict                    cy_stations
    cdef object                  cy_traveltime_inventory
    cdef fields.ScalarField3D    cy_grid
    cdef fields.ScalarField3D    cy_pwave_velocity
    cdef fields.ScalarField3D    cy_swave_velocity
    cdef dict                    cy_arrivals
    cdef dict                    cy_traveltimes
    cdef dict                    cy_residual_rvs
    cdef constants.REAL_t        cy_alpha
    # NLL-style additions
    cdef dict                    cy_pick_errors
    cdef constants.REAL_t        cy_default_pick_error
    cdef constants.REAL_t        cy_variance_floor
    cdef constants.REAL_t        cy_edt_exponent
    cdef constants.BOOL_t        cy_edt_ot_wt
    cdef constants.REAL_t        cy_edt_ot_wt_floor
    cdef object                  cy_locate_seed
    # flattened per-arrival workspace (rebuilt by _prepare_workspace)
    cdef list                    cy_keys
    cdef list                    cy_tt_fields
    cdef constants.REAL_t[:]     cy_obs
    cdef constants.REAL_t[:]     cy_sigma
    cdef constants.REAL_t[:]     cy_tt_work
    cdef constants.REAL_t[:]     cy_ot_work
    cdef object                  cy_edge_axes

    cpdef constants.BOOL_t add_arrivals(EQLocator self, dict arrivals)
    cpdef constants.BOOL_t add_residual_rvs(EQLocator self, dict residua_rvs)
    cpdef constants.BOOL_t clear_arrivals(EQLocator self)
    cpdef constants.BOOL_t clear_residual_rvs(EQLocator self)
    cpdef constants.BOOL_t read_traveltimes(
        EQLocator self,
        constants.REAL_t[:] min_coords=*,
        constants.REAL_t[:] max_coords=*
    )
    cpdef constants.REAL_t log_likelihood(
        EQLocator self,
        constants.REAL_t[:] model
    )

    cpdef constants.REAL_t rms(EQLocator self, constants.REAL_t[:] hypocenter)
    cpdef constants.BOOL_t add_stations(EQLocator self, dict stations)
    cpdef constants.BOOL_t add_pick_errors(EQLocator self, dict pick_errors)
    cpdef constants.BOOL_t _prepare_workspace(EQLocator self)
    cdef int _fill_traveltimes(EQLocator self, constants.REAL_t[:] hypo_xyz)
    cdef constants.REAL_t _effective_exponent(EQLocator self, int n)
    cpdef constants.REAL_t edt_log_likelihood(
        EQLocator self,
        constants.REAL_t[:] hypo_xyz
    )
    cpdef constants.REAL_t edt(EQLocator self, constants.REAL_t[:] hypo_xyz)
    cpdef constants.REAL_t origin_time(EQLocator self, constants.REAL_t[:] hypo_xyz)
    cpdef np.ndarray locate(
        EQLocator self,
        np.ndarray initial,
        np.ndarray delta,
        constants.REAL_t alpha=*,
        str method=*,
        np.ndarray bounds=*
    )
