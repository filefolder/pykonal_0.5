# distutils: language=c++
"""
This module provides three classes (:class:`Field3D <pykonal.fields.Field3D>`,
:class:`ScalarField3D <pykonal.fields.ScalarField3D>`, and
:class:`VectorField3D <pykonal.fields.VectorField3D>`). 
:class:`Field3D <pykonal.fields.Field3D>` is a base class intended to
be used only as an abstraction. :class:`ScalarField3D <pykonal.fields.ScalarField3D>`
and :class:`VectorField3D <pykonal.fields.VectorField3D>` both inherit
from :class:`Field3D <pykonal.fields.Field3D>` and add important
functionality that make the classes useful. They are primarily intended
for storing and interpolating data, and, in the case of scalar fields,
computing the gradient.

This module also provides an I/O function
(:func:`read_hdf <pykonal.fields.read_hdf>`) to read data that was saved using
the :func:`to_hdf <pykonal.fields.Field3D.to_hdf>` method of the above
classes.

.. autofunction:: pykonal.fields.load(path)
.. autofunction:: pykonal.fields.read_hdf(path)
"""

import warnings

# Third-party imports
import h5py
import numpy as np

# Local imports
from . import constants
from . import transformations

# Cython built-in imports.
cimport cython
from libc.math cimport sqrt, sin, isnan, NAN
from libcpp.vector cimport vector as cpp_vector
from libc.string cimport memcpy

# Third-party Cython imports
cimport numpy as np

# Local Cython imports
from . cimport constants


cdef class Field3D(object):
    """
    Base class for representing generic 3D fields.
    """

    def __init__(self, coord_sys="cartesian"):
        self.cy_coord_sys = coord_sys

    @property
    def coord_sys(self):
        """
        [*Read only*, :class:`str`] Coordinate system of grid on which
        field data is represented {"Cartesian", "spherical"}.
        """
        return (self.cy_coord_sys)

    @property
    def field_type(self):
        return (self.cy_field_type)

    @property
    def iax_isnull(self):
        """
        [*Read only*, :class:`numpy.ndarray`\ (shape=(3,), dtype=numpy.bool)]
        Array of booleans indicating whether each axis is null. The
        axis with only one layer of nodes in 2D problems will be null.
        """
        arr = np.array(
            [self.cy_iax_isnull[iax] for iax in range(3)],
            dtype=constants.DTYPE_BOOL
        )
        return (arr)

    @property
    def iax_isperiodic(self):
        """
        [*Read only*, :class:`numpy.ndarray`\ (shape=(3,), dtype=numpy.bool)]
        Array of booleans indicating whether each axis is periodic. In
        practice, only the azimuthal (:math:`\phi`) axis in spherical
        coordinates will ever be periodic.
        """
        arr = np.array(
            [self.cy_iax_isperiodic[iax] for iax in range(3)],
            dtype=constants.DTYPE_BOOL
        )
        return (arr)

    @property
    def min_coords(self):
        """
        [*Read/Write*, :class:`numpy.ndarray`\ (shape=(3,), dtype=numpy.float)]
        Array specifying the lower bound of each axis. This attribute
        must be initialized by the user.

        :math:`\phi` coordinate must be in [-:math:`\pi`, :math:`\pi`) or
        [0, 2:math:`\pi`] for spherical coordinates.
        """
        return (np.asarray(self.cy_min_coords))

    @min_coords.setter
    def min_coords(self, value):
        if self.coord_sys == "spherical" and value[0] == 0:
            raise (ValueError("ρ must be > 0 for spherical coordinates."))
        if self.coord_sys == "spherical" and value[2] < -np.pi:
            raise (ValueError("φ must be in [-π,π) or [0,2π) for spherical coordinates."))
        self.cy_min_coords = np.asarray(value, dtype=constants.DTYPE_REAL)
        self._update_max_coords()
        self._update_iax_isperiodic()
        self._invalidate_geometry()

    cdef constants.BOOL_t _invalidate_geometry(Field3D self):
        """
        Drop every cache derived from the grid geometry. Called by the
        min_coords / node_intervals / npts setters so a field can never be
        left describing one grid with caches built for another.
        """
        self.cy_nodes               = None
        self.cy_gradient            = None
        self.cy_norm_compact_isset  = False
        self.cy_step_size_isset     = False
        return True

    @property
    def max_coords(self):
        """
        [*Read only*, :class:`numpy.ndarray`\ (shape=(3,), dtype=numpy.float)]
        Array specifying the upper bound of each axis.
        """
        return np.asarray(self.cy_max_coords)

    @property
    def node_intervals(self):
        """
        [*Read/Write*, :class:`numpy.ndarray`\ (shape=(3,), dtype=numpy.float)]
        Array of node intervals along each axis. This attribute must be
        initialized by the user.
        """
        return np.asarray(self.cy_node_intervals)

    @node_intervals.setter
    def node_intervals(self, value):
        value = np.asarray(value, dtype=constants.DTYPE_REAL)
        if np.any(value <= 0):
            raise ValueError("All node intervals must be > 0")

        self.cy_node_intervals = value
        self._update_max_coords()
        self._update_iax_isperiodic()
        self._invalidate_geometry()

    @property
    def nodes(self):
        """
        [*Read only*, :class:`numpy.ndarray`\ (shape=(N0,N1,N2,3), dtype=numpy.float)]
        Array specifying the grid-node coordinates.
        """
        # cy_nodes is declared as np.ndarray (an object type) in
        # fields.pxd, so it defaults to None -- check against None rather
        # than try/except AttributeError.
        if self.cy_nodes is None:
            nodes = [
                np.linspace(
                    self.min_coords[idx],
                    self.max_coords[idx],
                    self.npts[idx],
                    dtype=constants.DTYPE_REAL
                )
                for idx in range(3)
            ]
            nodes = np.meshgrid(*nodes, indexing="ij")
            nodes = np.stack(nodes)
            nodes = np.moveaxis(nodes, 0, -1)
            self.cy_nodes = nodes
        return self.cy_nodes

    @property
    def norm(self):
        """
        [*Read-only*, numpy.ndarray(shape=(N0,N1,N2,3), dtype=numpy.float)] 4D array of scaling
        factors for gradient operator.
        """

        try:
            return np.asarray(self.cy_norm)
        except AttributeError:
            norm = np.tile(
                self.node_intervals,
                np.append(self.npts, 1)
            )
            if self.coord_sys == "spherical":
                norm[..., 1] *= self.nodes[..., 0]
                norm[..., 2] *= self.nodes[..., 0]
                norm[..., 2] *= np.sin(self.nodes[..., 1])
            self.cy_norm = norm
        return np.asarray(self.cy_norm)

    cdef constants.REAL_t[:,:,:] _norm_compact(Field3D self):
        """
        Gradient scale factors on a compact (N0, N1, 3) grid.

        The full `norm` array is (N0, N1, N2, 3), but its values do not
        depend on the third index: on a Cartesian grid the factors are just
        node_intervals, and on a spherical grid they are
        (dr, r*dtheta, r*sin(theta)*dphi) — functions of i0 and i1 only.
        Storing the N2 copies costs 3*N^3*8 bytes (192 MB at 200^3, roughly
        half the solver's footprint) and streams a redundant array through
        cache in the FMM inner loop.

        The multiplication ORDER below matches the `norm` property exactly
        so the two agree bit for bit; folding r and sin(theta) into a single
        product instead changes the last ulp.

        * CAN NOT BE CALLED BEFORE npts IS SET *
        """
        cdef Py_ssize_t n0, n1

        if self.cy_norm_compact_isset:
            return self.cy_norm_compact

        n0 = <Py_ssize_t>self.cy_npts[0]
        n1 = <Py_ssize_t>self.cy_npts[1]
        nc = np.tile(self.node_intervals, (n0, n1, 1))
        if self.cy_coord_sys == "spherical":
            rho = np.linspace(self.min_coords[0], self.max_coords[0], n0)
            theta = np.linspace(self.min_coords[1], self.max_coords[1], n1)
            nc[..., 1] *= rho[:, None]
            nc[..., 2] *= rho[:, None]
            nc[..., 2] *= np.sin(theta)[None, :]
        self.cy_norm_compact = nc
        self.cy_norm_compact_isset = True
        return self.cy_norm_compact

    @property
    def npts(self):
        """
        [*Read/Write*, :class:`numpy.ndarray`\ (shape=(3,), dtype=numpy.int)]
        Array specifying the number of nodes along each axis. This
        attribute must be initialized by the user.
        """
        return np.asarray(self.cy_npts)

    @npts.setter
    def npts(self, value):
        self.cy_npts = np.asarray(value, dtype=constants.DTYPE_UINT)
        self._update_max_coords()
        self._update_iax_isnull()
        self._update_iax_isperiodic()
        self._invalidate_geometry()        

    @property
    def step_size(self):
        """
        [*Read only*, :class:`float`] Step size used for ray tracing.
        """
        # cy_step_size is declared as a primitive constants.REAL_t (C
        # double) in fields.pxd, not an object type -- primitive cdef
        # attributes do NOT default to None/AttributeError the way typed
        # object attributes do; an unset one simply holds whatever
        # uninitialized memory it happens to contain. A try/except
        # AttributeError check here would silently never trigger and could
        # return garbage on first access, so an explicit boolean flag is
        # used instead to track whether the value has actually been
        # computed yet.
        if not self.cy_step_size_isset:
            norm = np.asarray(self._norm_compact())[..., ~self.iax_isnull]
            self.cy_step_size = norm[~np.isclose(norm, 0)].min() / 4
            self.cy_step_size_isset = True
        return self.cy_step_size


    cdef constants.BOOL_t _update_iax_isperiodic(Field3D self):
        if self.cy_coord_sys == "spherical":
            self.cy_iax_isperiodic[2] = np.isclose(
                self.cy_max_coords[2]+self.cy_node_intervals[2]-self.cy_min_coords[2],
                2 * np.pi
            )
        return True


    cdef constants.BOOL_t _update_iax_isnull(Field3D self):
        cdef Py_ssize_t       iax

        for iax in range(3):
            self.cy_iax_isnull[iax] = True if self.cy_npts[iax] == 1 else False
        return True


    cdef constants.BOOL_t _update_max_coords(Field3D self):
        cdef Py_ssize_t       iax
        cdef constants.REAL_t dx

        self.cy_norm_compact_isset = False
        self.cy_step_size_isset = False   

        for iax in range(3):
            dx = self.cy_node_intervals[iax] * <constants.REAL_t>(<Py_ssize_t>self.cy_npts[iax] - 1)
            self.cy_max_coords[iax] = self.cy_min_coords[iax] + dx

        if self.cy_coord_sys == "spherical":
            if self.cy_min_coords[2] >= 0 and self.cy_max_coords[2] > 2*np.pi:
                raise(ValueError("Phi coordinates must be in [-π, π) or [0, 2π)."))
            elif self.cy_min_coords[2] < 0 and self.cy_max_coords[2] > np.pi:
                raise(ValueError("Phi coordinates must be in [-π, π) or [0, 2π)."))

        return True


    def savez(self, path):
        """
        savez(self, path)

        Save the field to disk using numpy.savez.

        :param path: Path to output file.
        :type path: str

        :return: Returns True upon successful execution.
        :rtype: bool
        """

        warning_message = "The savez() method is deprecated and will be removed"\
            " from future versions of PyKonal. Use pykonal.fields.Field3D.to_hdf()"\
            " instead."
        warnings.warn(warning_message, DeprecationWarning)

        np.savez_compressed(
            path,
            min_coords=self.min_coords,
            node_intervals=self.node_intervals,
            npts=self.npts,
            values=self.values,
            coord_sys=[self.coord_sys]
        )
        return True


    cpdef constants.BOOL_t to_hdf(
        Field3D self,
        str path,
        str key=None,
        constants.BOOL_t overwrite=False,
        constants.BOOL_t compress=True,
        constants.BOOL_t low_precision=True):
        """
        to_hdf(self, path, key=None, overwrite=False, compress=True, low_precision=True)

        Save the field to an HDF5 file.

        :param compress: If True, write 'values' with gzip compression.
                         Field data (especially traveltime/velocity fields)
                         tends to be spatially smooth and compresses well.
        :type compress: bool
        :param low_precision: If True, write `values` as float32 rather
                              than float64. This halves storage for a
                              negligible precision loss (~1e-7 relative),
                              which is far below the physical precision of
                              traveltime/velocity data. `min_coords`,
                              `node_intervals`, and `npts` are always
                              written at full precision since they are
                              used for grid indexing.
        :type low_precision: bool
        """

        with h5py.File(path, mode="a") as f5:
            if key is not None:
                if key in f5 and overwrite is True:
                    del (f5[key])
                group = f5.create_group(key)

            else:
                group = f5["/"]

            group.attrs["coord_sys"] = self.coord_sys
            group.attrs["field_type"] = self.field_type

            for attr in ("min_coords", "node_intervals", "npts"):
                if attr in group and overwrite is True:
                    del (group[attr])
                group.create_dataset(attr, data=getattr(self, attr))

            if "values" in group and overwrite is True:
                del (group["values"])

            values = self.values
            if low_precision:
                values = values.astype(np.float32)

            kwargs = {}
            if compress:
                kwargs["compression"] = "gzip"
                kwargs["compression_opts"] = 4

            group.create_dataset("values", data=values, **kwargs)

        return True


    def transform_coordinates(self, coord_sys, origin, force_phi_positive=False):
        """
        transform_coordinates(self, coord_sys, origin)

        Transform node coordinates to a new frame of reference.

        :param coord_sys: Coordinate system to transform to
                          ("*spherical*", or "*Cartesian*")
        :type coord_sys: str
        :param origin: Coordinates of the origin of the new frame with
                       respect to the old frame of reference.
        :type origin: tuple(float, float, float)
        :param force_phi_positive: Force phi to be in [0, 2:pi)
                                   for output spherical coordinates.
        :type force_phi_positive: bool
        :return: Node coordinates in new frame of reference.
        :rtype: numpy.ndarray(shape=(N0,N1,N2,3), dtype=numpy.float)
        """
        if self.coord_sys == "spherical" and coord_sys.lower() == "spherical":
            return (
                transformations.sph2sph(
                    self.nodes,
                    origin,
                    force_phi_positive=force_phi_positive
                )
            )
        elif self.coord_sys == "cartesian" and coord_sys.lower() == "spherical":
            return (transformations.xyz2sph(self.nodes, origin))
        elif self.coord_sys == "spherical" and coord_sys.lower() == "cartesian":
            return (transformations.sph2xyz(self.nodes, origin))
        elif self.coord_sys == "cartesian" and coord_sys.lower() == "cartesian":
            return (transformations.xyz2xyz(self.nodes, origin))
        else:
            raise (NotImplementedError())




cdef class ScalarField3D(Field3D):
    """
    Class for representing 3D scalar fields.
    """
    def __init__(self, coord_sys="cartesian"):
        super(ScalarField3D, self).__init__(coord_sys=coord_sys)
        self.cy_field_type = "scalar"

    @property
    def gradient(self):
        """
        [*Read only*, numpy.ndarray(shape=(N0,N1,N2,3), dtype=numpy.float)]
        Gradient of the field.
        """
        # cy_gradient is a typed cdef attribute (declared in fields.pxd),
        # so it defaults to None rather than being unset/raising
        # AttributeError -- check against None, matching the pattern used
        # for cy_trial/cy_traveltime in solver.pyx.
        if self.cy_gradient is None:
            if self.coord_sys == "cartesian":
                self.cy_gradient = self._gradient_of_cartesian()
            elif self.coord_sys == "spherical":
                self.cy_gradient = self._gradient_of_spherical()
        return self.cy_gradient

    @property
    def values(self):
        """
        [*Read/Write*, numpy.ndarray(shape=(N0,N1,N2), dtype=numpy.float)]
        Value of the field at each grid node.
        """
        try:
            return (np.asarray(self.cy_values))
        except AttributeError:
            self.cy_values = np.full(self.npts, fill_value=np.nan)
        return np.asarray(self.cy_values)

    @values.setter
    def values(self, value):
        values = np.asarray(value, dtype=constants.DTYPE_REAL)
        if not np.all(values.shape == self.npts):
            raise (ValueError("Shape of values does not match npts attribute."))
        # Write into the existing buffer rather than rebinding whenever one
        # of a compatible shape already exists. Anything else holding a view
        # of this field's values -- most importantly heapq.Heap, which sorts
        # the FMM narrow band by traveltime -- keeps seeing live data.
        # An unset typed memoryview raises AttributeError on access rather
        # than comparing equal to None, hence try/except.
        try:
            reuse = (
                    self.cy_values.shape[0] == values.shape[0]
                and self.cy_values.shape[1] == values.shape[1]
                and self.cy_values.shape[2] == values.shape[2]
            )
        except AttributeError:
            reuse = False

        if reuse:
            np.asarray(self.cy_values)[...] = values
        else:
            self.cy_values = values
        self.cy_gradient = None


    cpdef np.ndarray[constants.REAL_t, ndim=1] resample(
        ScalarField3D self, constants.REAL_t[:,:] points, constants.REAL_t null=np.nan):
        """
        resample(self, points, null=numpy.nan)

        Resample the field at an arbitrary set of points using
        trilinear interpolation.

        :param points: Points at which to resample the field.
        :type points: numpy.ndarray(shape=(N,3), dtype=numpy.float)
        :param null: Default (null) value to return for points lying
                     outside the interpolation domain.
        :type null: float
        :return: Resampled field values.
        :rtype: numpy.ndarray(shape=(N,), dtype=numpy.float)
        """
        cdef Py_ssize_t                           idx
        cdef np.ndarray[constants.REAL_t, ndim=1] resampled # Using a MemoryViewBuffer might make this faster.

        resampled = np.empty(points.shape[0], dtype=constants.DTYPE_REAL)

        for idx in range(points.shape[0]):
            resampled[idx] = self.value(points[idx], null=null)

        return resampled


    cpdef np.ndarray[constants.REAL_t, ndim=2] trace_ray(
            ScalarField3D self,
            constants.REAL_t[:] end):
        """
        trace_ray(self, end)

        Trace the ray ending at *end* (given in the
        same coordinate system as self.coord_sys.)

        This method traces the ray that ends at *end* in reverse
        direction by taking small steps along the path of steepest
        descent. The resulting ray is reversed before being returned,
        so it is in the normal forward-time orientation.

        :param end: Coordinates of the ray's end point.
        :type end: numpy.ndarray(shape=(3,), dtype=numpy.float)

        :return: The ray path ending at *end*.
        :rtype:  numpy.ndarray(shape=(N,3), dtype=numpy.float)
        """

        cdef cpp_vector[constants.REAL_t]         ray
        cdef constants.REAL_t                     norm, step_size, value, value_1back
        cdef constants.REAL_t[3]                  gg, pt_local
        cdef constants.REAL_t[:]                  point
        cdef Py_ssize_t                           idx
        cdef str                                  coord_sys
        cdef np.ndarray[constants.REAL_t, ndim=1] ray_np
        cdef VectorField3D                        grad
        cdef bint                                 is_spherical

        cdef Py_ssize_t                           max_steps
        max_steps = 100000

        coord_sys = self.coord_sys
        is_spherical = (coord_sys == "spherical")
        step_size = self.step_size
        grad      = self.gradient

        for idx in range(3):
            ray.push_back(end[idx])

        value = np.inf

        while True:
            if ray.size() // 3 > max_steps:
                return None # runaway condition
            point = <constants.REAL_t[:3]>&ray[ray.size()-3]
            try:
                value_1back = value
                value = self.value(point)
                if value >= value_1back or isnan(value):
                    ray.resize(ray.size()-3) # drop bad pt
                    break
            except ValueError:
                return None
            grad.value_c(&point[0], gg, NAN)
            norm = sqrt(gg[0]*gg[0] + gg[1]*gg[1] + gg[2]*gg[2])
            if isnan(norm):
                raise (ValueError("Encountered NaN gradient."))
            for idx in range(3):
                gg[idx] /= norm
            if is_spherical:
                gg[1] /= point[0]
                gg[2] /= point[0] * sin(point[1])
            for idx in range(3):
                pt_local[idx] = point[idx]      # all reads, before anything moves
            for idx in range(3):
                ray.push_back(pt_local[idx] - step_size * gg[idx])   # all writes

        if ray.size() < 4:
            return None  # degenerate    

        ray_np = np.empty(ray.size(), dtype=constants.DTYPE_REAL)
        memcpy(&ray_np[0], ray.data(), ray.size() * sizeof(constants.REAL_t))
        return np.flipud(ray_np.reshape(-1,3))


    @cython.initializedcheck(False)
    cpdef constants.REAL_t value(
        ScalarField3D self,
        constants.REAL_t[:] point,
        constants.REAL_t null=np.nan
    ):
        cdef constants.REAL_t[3]         delta, idx
        cdef constants.REAL_t            f000, f100, f110, f101, f111, f010, f011, f001
        cdef constants.REAL_t            f00, f10, f01, f11
        cdef constants.REAL_t            f0, f1
        cdef constants.REAL_t            f
        cdef Py_ssize_t[3][2]            ii
        cdef Py_ssize_t                  i1, i2, i3, iax, di1, di2, di3, n
        cdef constants.REAL_t            pt_iax
        cdef constants.REAL_t            span

        for iax in range(3):

            pt_iax = point[iax]
            n      = <Py_ssize_t>self.cy_npts[iax]

            if self.cy_iax_isperiodic[iax]:
                # Wrap the coordinate into [min_coords, max_coords) before
                # computing idx/delta below, so points passed in slightly
                # outside the nominal range (e.g. phi = -1e-9, or
                # phi = 2*pi + 0.1) are handled correctly instead of being
                # spuriously rejected or mis-interpolated at the seam.
                span = self.cy_max_coords[iax] + self.cy_node_intervals[iax] - self.cy_min_coords[iax]
                pt_iax = self.cy_min_coords[iax] + (
                    (pt_iax - self.cy_min_coords[iax]) % span
                )
            elif (
                (pt_iax < self.cy_min_coords[iax] or pt_iax > self.cy_max_coords[iax])
                and not self.cy_iax_isnull[iax]
            ):
                return null

            idx[iax]   = (pt_iax - self.cy_min_coords[iax]) / self.cy_node_intervals[iax]
            if self.cy_iax_isnull[iax]:
                ii[iax][0] = 0
                ii[iax][1] = 0

            else:
                ii[iax][0]  = <Py_ssize_t>idx[iax]
                ii[iax][1]  = ((ii[iax][0] + 1) % n)
                # **** double check still in bounds (RCP)
                if ii[iax][0] < 0 or ii[iax][0] >= n or ii[iax][1] < 0 or ii[iax][1] >= n:
                    return null

            # idx[iax] % 1 here uses C fmod semantics (sign follows the
            # dividend), but idx[iax] is guaranteed non-negative at this
            # point: non-periodic out-of-range values already returned
            # above, and periodic values were wrapped into
            # [min_coords, max_coords) so (pt_iax - min_coords) >= 0.
            delta[iax] = idx[iax] % 1

        f000    = self.cy_values[ii[0][0], ii[1][0], ii[2][0]]
        f100    = self.cy_values[ii[0][1], ii[1][0], ii[2][0]]
        f110    = self.cy_values[ii[0][1], ii[1][1], ii[2][0]]
        f101    = self.cy_values[ii[0][1], ii[1][0], ii[2][1]]
        f111    = self.cy_values[ii[0][1], ii[1][1], ii[2][1]]
        f010    = self.cy_values[ii[0][0], ii[1][1], ii[2][0]]
        f011    = self.cy_values[ii[0][0], ii[1][1], ii[2][1]]
        f001    = self.cy_values[ii[0][0], ii[1][0], ii[2][1]]
        f00     = f000 + (f100 - f000) * delta[0]
        f10     = f010 + (f110 - f010) * delta[0]
        f01     = f001 + (f101 - f001) * delta[0]
        f11     = f011 + (f111 - f011) * delta[0]
        f0      = f00  + (f10  - f00)  * delta[1]
        f1      = f01  + (f11  - f01)  * delta[1]
        f       = f0   + (f1   - f0)   * delta[2]
        return f



    cpdef VectorField3D _gradient_of_cartesian(ScalarField3D self):
        """
        The gradient of a field represented on a Cartesian grid.
        """
        gg = np.gradient(
            self.values,
            *[
                self.node_intervals[iax]
                for iax in range(3) if not self.iax_isnull[iax]
            ],
            axis=[
                iax
                for iax in range(3)
                if not self.iax_isnull[iax]
            ]
        )
        gg = np.moveaxis(np.stack(gg), 0, -1)
        for iax in range(3):
            if self.iax_isnull[iax]:
                gg = np.insert(gg, iax, np.zeros(self.npts), axis=-1)
        grad = VectorField3D(coord_sys=self.coord_sys)
        grad.min_coords = self.min_coords
        grad.node_intervals = self.node_intervals
        grad.npts = self.npts
        grad.values = gg
        return grad


    cpdef VectorField3D _gradient_of_spherical(ScalarField3D self):
        """
        The gradient of a field represented on a spherical grid
        """
        grid       = self.nodes
        d0, d1, d2 = self.node_intervals
        n0, n1, n2 = np.asarray(self.npts).astype(int)

        # DEBUG
        _vshape = np.asarray(self.values).shape
        _npts   = tuple(np.asarray(self.npts).astype(int).tolist())
        if tuple(_vshape) != _npts:
            raise ValueError(
                "values.shape=%s disagrees with npts=%s" % (tuple(_vshape), _npts)
            )

        for iax, n in enumerate((n0, n1, n2)):
            if not self.iax_isnull[iax] and n < 3:
                raise ValueError(
                    f"Cannot compute spherical gradient along axis {iax}: "
                    f"npts={n} but at least 3 points are required for the "
                    f"finite-difference stencil used here (or the axis "
                    f"must be null, i.e. npts=1)."
                )

        if not self.iax_isnull[0]:
            g0_lower = (
                (
                        self.values[2]
                    - 4*self.values[1]
                    + 3*self.values[0]
                ) / (2*d0)
            ).reshape(1, n1, n2)
            g0_interior = (self.values[2:] - self.values[0:n0-2]) / (2*d0)
            g0_upper = (
                (
                        self.values[n0-3]
                    - 4*self.values[n0-2]
                    + 3*self.values[n0-1]
                ) / (2*d0)
            ).reshape(1, n1, n2)
            g0 = np.concatenate([g0_lower, g0_interior, g0_upper], axis=0)
        else:
            g0 = np.zeros((n0, n1, n2))

        if not self.iax_isnull[1]:
            g1_lower = (
                (
                        self.values[:,2]
                    - 4*self.values[:,1]
                    + 3*self.values[:,0]
                ) / (2*grid[:,0,:,0]*d1)
            ).reshape(n0, 1, n2)
            g1_interior = (
                  self.values[:,2:]
                - self.values[:,0:n1-2]
            ) / (2*grid[:,1:n1-1,:,0]*d1)
            g1_upper = (
                (
                        self.values[:,n1-3]
                    - 4*self.values[:,n1-2]
                    + 3*self.values[:,n1-1]
                ) / (2*grid[:,n1-1,:,0]*d1)
            ).reshape(n0, 1, n2)
            g1 = np.concatenate([g1_lower, g1_interior, g1_upper], axis=1)
        else:
            g1 = np.zeros((n0, n1, n2))

        if not self.iax_isnull[2]:
            g2_lower = (
                  (
                        self.values[:,:,2]
                    - 4*self.values[:,:,1]
                    + 3*self.values[:,:,0]
                ) / (2*grid[:,:,0,0]*np.sin(grid[:,:,0,1])*d2)
            ).reshape(n0, n1, 1)
            g2_interior = (
                  self.values[:,:,2:]
                - self.values[:,:,0:n2-2]
            ) / (2*grid[:,:,1:n2-1,0]*np.sin(grid[:,:,1:n2-1,1])*d2)
            g2_upper = (
                (
                        self.values[:,:,n2-3]
                    - 4*self.values[:,:,n2-2]
                    + 3*self.values[:,:,n2-1]
                ) / (2*grid[:,:,n2-1,0]*np.sin(grid[:,:,n2-1,1])*d2)
            ).reshape(n0, n1, 1)
            g2 = np.concatenate([g2_lower, g2_interior, g2_upper], axis=2)
        else:
            g2 = np.zeros((n0, n1, n2))

        gg = np.moveaxis(np.stack([g0, g1, g2]), 0, -1)
        grad = VectorField3D(coord_sys=self.coord_sys)
        grad.min_coords = self.min_coords
        grad.node_intervals = self.node_intervals
        grad.npts = self.npts
        grad.values = gg
        return grad



cdef class VectorField3D(Field3D):
    """
    Class for representing 3D vector fields
    """

    def __init__(self, coord_sys="cartesian"):
        super(VectorField3D, self).__init__(coord_sys=coord_sys)
        self.cy_field_type = "vector"

    @property
    def values(self):
        """
        [*Read/Write*, numpy.ndarray(shape=(N0,N1,N2,3), dtype=numpy.float)]
        Value of the field at each grid node.
        """
        try:
            return np.asarray(self.cy_values)
        except AttributeError:
            self.cy_values = np.full(self.npts, fill_value=np.nan)
        return np.asarray(self.cy_values)

    @values.setter
    def values(self, value):
        values = np.asarray(value)
        if not np.all(values.shape[:3] == self.npts):
            raise (ValueError("Shape of values does not match npts attribute."))
        self.cy_values = np.asarray(value)

    @cython.initializedcheck(False)
    cdef void value_c(
        VectorField3D self,
        constants.REAL_t* point,
        constants.REAL_t* out,
        constants.REAL_t null
    ) noexcept nogil:
        """
        value_c(self, point, out, null)

        C-only trilinear interpolation of the field at *point*, writing the
        three components into the caller-supplied buffer *out*.

        This is the hot path: it takes and returns raw pointers so that no
        ndarray is allocated per call. `value()` is a thin Python-facing
        wrapper around it and is the one to call from Python.

        :param point: Pointer to 3 contiguous coordinates.
        :param out:   Pointer to 3 contiguous doubles to write into. Every
                      component is set to *null* if *point* lies outside the
                      interpolation domain.
        :param null:  Value written to every component of *out* when *point*
                      is outside the domain.
        """
        cdef constants.REAL_t[3]         delta, idx
        cdef constants.REAL_t            f000, f100, f110, f101, f111, f010, f011, f001
        cdef constants.REAL_t            f00, f10, f01, f11
        cdef constants.REAL_t            f0, f1
        cdef constants.REAL_t            f
        cdef Py_ssize_t[3][2]            ii
        cdef Py_ssize_t                  iax, n
        cdef constants.REAL_t            pt_iax
        cdef constants.REAL_t            span

        for iax in range(3):

            pt_iax = point[iax]
            n      = <Py_ssize_t>self.cy_npts[iax]

            if self.cy_iax_isperiodic[iax]:
                # Wrap periodic (phi) coordinates into
                # [min_coords, max_coords) instead of rejecting/mis-binning
                # points just outside the nominal range due to floating
                # point error or caller convention.
                #
                # The `+ span` guard is REQUIRED. With cdivision=True, `%`
                # on C doubles compiles to fmod, whose sign follows the
                # dividend, so a negative phi would NOT be wrapped: it would
                # either be rejected (phi = -54 -> null) or, worse, truncate
                # to a valid-looking index with a negative delta and
                # silently extrapolate (phi = -0.5). The pre-existing
                # `value()` avoided this only by accident, because
                # `self.min_coords[iax]` returned a numpy scalar and so used
                # NumPy's floored modulo.
                span = self.cy_max_coords[iax] + self.cy_node_intervals[iax] - self.cy_min_coords[iax]
                pt_iax = (pt_iax - self.cy_min_coords[iax]) % span
                if pt_iax < 0:
                    pt_iax = pt_iax + span
                pt_iax = self.cy_min_coords[iax] + pt_iax
            elif (
                (pt_iax < self.cy_min_coords[iax] or pt_iax > self.cy_max_coords[iax])
                and not self.cy_iax_isnull[iax]):
                # Restored bounds check (was previously commented out
                # entirely). Without this, out-of-range points fall through
                # to raw array indexing below with no guarantee the
                # resulting indices are in bounds -- in the worst case this
                # reads outside the underlying buffer. Now returns `null`
                # for every component, consistent with ScalarField3D.value().
                out[0] = null
                out[1] = null
                out[2] = null
                return

            idx[iax]   = (pt_iax - self.cy_min_coords[iax]) / self.cy_node_intervals[iax]
            if self.cy_iax_isnull[iax]:
                ii[iax][0] = 0
                ii[iax][1] = 0
            else:
                ii[iax][0]  = <Py_ssize_t>idx[iax]
                ii[iax][1]  = ((ii[iax][0] + 1) % n)
                if ii[iax][0] < 0 or ii[iax][0] >= n or ii[iax][1] < 0 or ii[iax][1] >= n:
                    out[0] = null
                    out[1] = null
                    out[2] = null
                    return
            delta[iax] = idx[iax] % 1

        for iax in range(3):
            f000     = self.cy_values[ii[0][0], ii[1][0], ii[2][0], iax]
            f100     = self.cy_values[ii[0][1], ii[1][0], ii[2][0], iax]
            f110     = self.cy_values[ii[0][1], ii[1][1], ii[2][0], iax]
            f101     = self.cy_values[ii[0][1], ii[1][0], ii[2][1], iax]
            f111     = self.cy_values[ii[0][1], ii[1][1], ii[2][1], iax]
            f010     = self.cy_values[ii[0][0], ii[1][1], ii[2][0], iax]
            f011     = self.cy_values[ii[0][0], ii[1][1], ii[2][1], iax]
            f001     = self.cy_values[ii[0][0], ii[1][0], ii[2][1], iax]
            f00      = f000 + (f100 - f000) * delta[0]
            f10      = f010 + (f110 - f010) * delta[0]
            f01      = f001 + (f101 - f001) * delta[0]
            f11      = f011 + (f111 - f011) * delta[0]
            f0       = f00  + (f10  - f00)  * delta[1]
            f1       = f01  + (f11  - f01)  * delta[1]
            f        = f0   + (f1   - f0)   * delta[2]
            out[iax] = f


    cpdef np.ndarray[constants.REAL_t, ndim=1] value(
        VectorField3D self,
        constants.REAL_t[:] point,
        constants.REAL_t null=np.nan
    ):
        """
        value(self, point, null=numpy.nan)

        Interpolate the field at *point* using trilinear interpolation.

        :param point: Coordinates of the point at which to interpolate
                      the field.
        :type point: numpy.ndarray(shape=(3,), dtype=numpy.float)
        :param null: Default (null) value to return (for every component)
                     if point lies outside the interpolation domain.
        :type null: float
        :return: Value of the field at *point*.
        :rtype: numpy.ndarray(shape=(3,), dtype=numpy.float)
        """
        cdef constants.REAL_t[3] ff

        self.value_c(&point[0], ff, null)
        return np.asarray(ff)



cpdef Field3D load(str path):
    """
    .. deprecated:: 0.3.2
       Use :func:`read_hdf` instead.

    Load field data from disk.

    :param path: Path to input file.
    :type path: str
    :return: A Field3D-derivative class initialized with data in *path*.
    :rtype: ScalarField3D or VectorField3D
    """
    warning_message = "The load() function is deprecated and will be removed"\
        " from future versions of PyKonal. Use pykonal.fields.read_hdf()"\
        " instead."
    warnings.warn(warning_message, DeprecationWarning)

    with np.load(path) as npz:
        coord_sys = str(npz["coord_sys"][0])
        if len(npz["values"].shape) == 4:
            field = VectorField3D(coord_sys=coord_sys)
        else:
            field = ScalarField3D(coord_sys=coord_sys)
        field.min_coords = npz["min_coords"]
        field.node_intervals = npz["node_intervals"]
        field.npts = npz["npts"]
        field.values = npz["values"]
    return field


def read_hdf(path, min_coords=None, max_coords=None):
    """
    Load field data from HDF5 file on disk.

    :param path: Path to input file.
    :type path: str
    :param min_coords: Minimum bounding coordinates to read. This is for
                       reading a limited portion of the file and should
                       be given in the same coordinates as self.coord_sys.
    :type min_coords: numpy.ndarray(shape=(3,), dtype=numpy.float), optional
    :param max_coords: Maximum bounding coordinates to read. This is for
                       reading a limited portion of the file and should
                       be given in the same coordinates as self.coord_sys.
    :type max_coords: numpy.ndarray(shape=(3,), dtype=numpy.float), optional
    :return: A Field3D-derivative class initialized with data in *path*.
    :rtype: ScalarField3D or VectorField3D
    """

    with h5py.File(path, mode="r") as f5:

        _coord_sys = f5.attrs["coord_sys"]
        _field_type = f5.attrs["field_type"]
        _min_coords = f5["min_coords"][:]
        _node_intervals = f5["node_intervals"][:]
        _npts = f5["npts"][:]

        if min_coords is not None:
            min_coords = np.array(min_coords)

        if max_coords is not None:
            max_coords = np.array(max_coords)

        if min_coords is not None and max_coords is not None:
            if np.any(min_coords >= max_coords):
                raise(ValueError("All values of min_coords must satisfy min_coords < max_coords."))

        if min_coords is not None:
            idx_start = (min_coords - _min_coords) / _node_intervals
            idx_start = np.floor(idx_start)
            idx_start = idx_start.astype(np.int32)
            idx_start = np.clip(idx_start, 0, _npts - 1)

        else:
            idx_start = np.array([0, 0, 0])

        if max_coords is not None:
            idx_end = (max_coords - _min_coords) / _node_intervals
            idx_end = np.ceil(idx_end) + 1
            idx_end = idx_end.astype(np.int32)
            idx_end = np.clip(idx_end, idx_start + 1, _npts)

        else:
            idx_end = _npts

        if _field_type == "scalar":
            field = ScalarField3D(coord_sys=_coord_sys)
        elif _field_type == "vector":
            field = VectorField3D(coord_sys=_coord_sys)
        else:
            raise (ValueError(f"Unrecognized field type: {_field_type}"))

        field.min_coords = _min_coords  +  idx_start * _node_intervals
        field.node_intervals = _node_intervals
        field.npts = idx_end - idx_start
        idxs = tuple(slice(idx_start[idx], idx_end[idx]) for idx in range(3))
        field.values = f5["values"][idxs].astype(constants.DTYPE_REAL)

    return field
