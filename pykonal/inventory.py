import h5py
import numpy as np
import os
import warnings

from . import constants
from . import fields


class TraveltimeInventory(object):

    def __init__(self, path, mode="r"):
        self._mode = mode
        self._path = path
        self._f5 = h5py.File(path, mode=mode)

    def __del__(self):
        self.f5.close()

    def __enter__(self):
        return (self)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__del__()

    @property
    def f5(self):
        return (self._f5)

    @property
    def mode(self):
        return (self._mode)

    @mode.setter
    def mode(self, value):
        self._mode = value
        self.f5.close()
        self._f5 = h5py.File(self.path, mode=value)

    @property
    def path(self):
        return (self._path)


    def add(self, field, key, station_coords=None, max_dist=None,
            mask=True, compress=True):
        """
        Store a traveltime field, by default distance-limited to nodes
        within *max_dist* (km) epicentral distance of the station.

        Parameters
        ----------
        field : ScalarField3D
            Traveltime field to store.
        key : str
            HDF5 group key, e.g. "NET/STA/PHASE".
        station_coords : array-like, recommended
            Station location in the field's coordinate system (e.g. the
            solver's src_loc). Strongly recommended whenever max_dist is
            used. If None, it is inferred as the grid node with the
            minimum traveltime (with a warning); inference can misbehave
            on unusual grids, so passing it explicitly is more robust.
            A sanity check warns if the given coordinates lie more than
            3 grid nodes from the field's traveltime minimum (i.e. they
            do not look like the source of this field), and raises if
            they fall outside the grid.
        max_dist : float or None
            Epicentral (horizontal / great-circle) distance cutoff in km.
            The stored grid is cropped to the bounding box of this radius,
            and (if *mask* is True) nodes beyond the radius are set to NaN
            so the corners of the box compress away and locators treat
            them as no-coverage. None disables distance limiting.
        mask : bool
            Mask nodes beyond max_dist with NaN (default True).
        compress : bool
            Store values chunked with gzip compression (default True).
            Traveltime fields are smooth and typically compress 2-4x on
            top of the distance-limiting savings.

        Coordinate conventions: "cartesian" fields are (x, y, z[depth]) in
        km with distance measured in the x-y plane; "spherical" fields are
        (r, theta[colatitude], phi[longitude]) with distance measured as
        great-circle arc length at the Earth's surface. The depth/radial
        axis is never cropped.
        """

        values = np.asarray(field.values, dtype=np.float64)
        min_coords = np.asarray(field.min_coords, dtype=np.float64)
        node_intervals = np.asarray(field.node_intervals, dtype=np.float64)
        npts = np.asarray(field.npts, dtype=np.int64)

        idx_start = np.array([0, 0, 0], dtype=np.int64)
        idx_end = npts.copy()

        if max_dist is not None:

            if station_coords is None:
                # fallback: the source node has the minimum traveltime.
                # Inference can glitch on unusual grids; prefer passing
                # station_coords explicitly (e.g. solver.src_loc).
                src_idx = np.unravel_index(np.nanargmin(values), values.shape)
                station_coords = min_coords + np.array(src_idx) * node_intervals
                warnings.warn(
                    f"{key}: station_coords not provided; inferred "
                    f"{np.round(station_coords, 4)} from the traveltime "
                    f"minimum. Pass station_coords explicitly for "
                    f"robustness.",
                    stacklevel=2
                )
            station_coords = np.asarray(station_coords, dtype=np.float64)

            # sanity check: traveltime at the station should be ~0. Compare
            # against the time to cross a few nodes at the local apparent
            # velocity implied by neighboring values.
            idx = np.rint(
                (station_coords - min_coords) / node_intervals
            ).astype(np.int64)
            if np.any(idx < 0) or np.any(idx >= npts):
                raise ValueError(
                    f"{key}: station_coords {station_coords} fall outside "
                    f"the grid."
                )
            # sanity: the claimed station location should coincide with
            # the field's traveltime minimum to within a few grid nodes.
            # Comparing node indices (rather than traveltime values) is
            # immune to the near-source interpolation error that leaves
            # the coarse source node with a small nonzero time.
            src_idx = np.array(
                np.unravel_index(np.nanargmin(values), values.shape)
            )
            offset = np.max(np.abs(idx - src_idx))
            if offset > 3:
                warnings.warn(
                    f"{key}: station_coords are {offset} grid nodes from "
                    f"the field's traveltime minimum (node {tuple(src_idx)} "
                    f"vs claimed {tuple(idx)}). The provided coordinates "
                    f"may not match this field's source; the distance "
                    f"limit may crop the wrong region.",
                    stacklevel=2
                )

            axes = [
                min_coords[iax] + np.arange(npts[iax]) * node_intervals[iax]
                for iax in range(3)
            ]

            if field.coord_sys == "cartesian":
                # crop x (axis 0) and y (axis 1) to +/- max_dist; keep z
                for iax in (0, 1):
                    lo = station_coords[iax] - max_dist
                    hi = station_coords[iax] + max_dist
                    idx_start[iax] = np.searchsorted(axes[iax], lo, "left")
                    idx_end[iax] = np.searchsorted(axes[iax], hi, "right")
            else:
                # spherical: angular radius, crop theta (axis 1) and
                # phi (axis 2); keep r (axis 0)
                dd = max_dist / constants.EARTH_RADIUS
                lo = station_coords[1] - dd
                hi = station_coords[1] + dd
                idx_start[1] = np.searchsorted(axes[1], lo, "left")
                idx_end[1] = np.searchsorted(axes[1], hi, "right")
                # widen the phi window by 1/sin(theta) at the extreme
                # colatitudes of the (cropped) theta range
                theta_lo = axes[1][idx_start[1]]
                theta_hi = axes[1][min(idx_end[1], npts[1]) - 1]
                sin_min = min(np.sin(theta_lo), np.sin(theta_hi))
                if sin_min * constants.EARTH_RADIUS > 1e-3:
                    dphi = dd / sin_min
                    lo = station_coords[2] - dphi
                    hi = station_coords[2] + dphi
                    idx_start[2] = np.searchsorted(axes[2], lo, "left")
                    idx_end[2] = np.searchsorted(axes[2], hi, "right")

            idx_start = np.clip(idx_start, 0, npts - 1)
            idx_end = np.clip(idx_end, idx_start + 1, npts)

            slices = tuple(
                slice(idx_start[iax], idx_end[iax]) for iax in range(3)
            )
            values = values[slices].copy()
            min_coords = min_coords + idx_start * node_intervals
            npts = idx_end - idx_start
            axes = [
                min_coords[iax] + np.arange(npts[iax]) * node_intervals[iax]
                for iax in range(3)
            ]

            if mask:
                if field.coord_sys == "cartesian":
                    dx = axes[0] - station_coords[0]
                    dy = axes[1] - station_coords[1]
                    dist = np.sqrt(
                        dx[:, None] ** 2 + dy[None, :] ** 2
                    )  # (nx, ny)
                    values[dist > max_dist, :] = np.nan
                else:
                    # great-circle distance on the sphere
                    lat0 = np.pi / 2 - station_coords[1]
                    lat = np.pi / 2 - axes[1]
                    dlon = axes[2] - station_coords[2]
                    a = (
                        np.sin((lat[:, None] - lat0) / 2) ** 2
                        + np.cos(lat0)
                        * np.cos(lat)[:, None]
                        * np.sin(dlon[None, :] / 2) ** 2
                    )
                    gc = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
                    dist = gc * constants.EARTH_RADIUS  # (ntheta, nphi)
                    values[:, dist > max_dist] = np.nan

        group = self.f5.create_group(key)
        group.attrs["coord_sys"] = field.coord_sys
        group.attrs["field_type"] = field.field_type
        if max_dist is not None:
            group.attrs["max_dist"] = max_dist
            group.attrs["station_coords"] = station_coords

        group.create_dataset("min_coords", data=min_coords)
        group.create_dataset("node_intervals", data=node_intervals)
        group.create_dataset("npts", data=npts)

        # write the actual times only as float32
        values = values.astype(np.float32)
        if compress:
            group.create_dataset(
                "values", data=values,
                chunks=True, compression="gzip", compression_opts=4,
                shuffle=True
            )
        else:
            group.create_dataset("values", data=values)

        return (True)


    def has(self, key):
        """True if a traveltime field is stored under *key* (str or tuple)."""
        if not isinstance(key, str):
            key = "/".join(key)
        return key in self.f5


    def keys(self):
        """List of stored 'NET/STA/PHASE' keys."""
        found = []
        self.f5.visit(
            lambda name: found.append(name)
            if name.count("/") == 2 and "values" not in name else None
        )
        # visit yields all paths; keep groups that hold a values dataset
        return [k for k in found if f"{k}/values" in self.f5]


    def ensure(self, requests, velocity_models, max_dist=None,
               mask=True, compress=True, solver_kwargs=None):
        """
        Compute-and-store any missing traveltime grids, so subsequent runs
        (and concurrent processes) find them precomputed.

        Parameters
        ----------
        requests : dict
            {("NET", "STA", "PHASE"): station_coords} for every key the
            upcoming inversion needs. Coordinates are in the velocity
            model's coordinate system.
        velocity_models : dict
            {"P": ScalarField3D or path-to-hdf, "S": ...}. Phases without
            a model cannot be computed and are reported as skipped.
        max_dist, mask, compress :
            Passed to add() for the newly computed grids.
        solver_kwargs : dict, optional
            Extra attributes to set on each PointSourceSolver.

        Returns
        -------
        dict with keys "present", "computed", "skipped" (lists of keys).

        Concurrency: this method does NOT lock. HDF5 enforces its own
        single-writer file locking at open time, so for cross-process
        safety use the module-level ensure_traveltimes(), which acquires
        an exclusive lock on <path>.lock BEFORE opening the file. Readers
        should hold a shared lock on the same lock file while the
        inventory is open (the SeisComP wrapper does this).
        """
        from . import solver as _solver

        if self.f5.mode == "r":
            raise IOError(
                "ensure() needs a writable inventory; open with mode='a'."
            )

        missing = {k: v for k, v in requests.items() if not self.has(k)}
        result = {
            "present": [k for k in requests if k not in missing],
            "computed": [], "skipped": []
        }
        if not missing:
            return result

        # resolve velocity models lazily (only if something is missing)
        vmodels = {}
        for phase, vm in velocity_models.items():
            vmodels[phase.upper()] = (
                fields.read_hdf(vm) if isinstance(vm, (str, os.PathLike))
                else vm
            )

        for key, coords in missing.items():
            if self.has(key):
                result["present"].append(key)
                continue
            phase = key[2].upper() if len(key) > 2 else "P"
            vmodel = vmodels.get(phase)
            if vmodel is None or coords is None:
                warnings.warn(
                    f"{'/'.join(key)}: cannot compute traveltimes "
                    f"({'no velocity model for phase ' + phase if vmodel is None else 'no station coordinates'}); "
                    f"skipping.",
                    stacklevel=2
                )
                result["skipped"].append(key)
                continue

            slv = _solver.PointSourceSolver(coord_sys=vmodel.coord_sys)
            slv.velocity.min_coords = vmodel.min_coords
            slv.velocity.node_intervals = vmodel.node_intervals
            slv.velocity.npts = vmodel.npts
            slv.velocity.values = vmodel.values
            if solver_kwargs:
                for attr, value in solver_kwargs.items():
                    setattr(slv, attr, value)
            slv.src_loc = np.asarray(coords, dtype=np.float64)
            slv.solve()
            self.add(
                slv.traveltime, "/".join(key),
                station_coords=slv.src_loc,
                max_dist=max_dist, mask=mask, compress=compress
            )
            result["computed"].append(key)
        self.f5.flush()

        return result


    def merge(self, paths, station_coords=None, max_dist=None,
              mask=True, compress=True):
        """
        station_coords: optional dict mapping (network, station) or
        "NET.STA" to coordinates in the fields' coordinate system,
        passed through to add() for robust distance limiting.
        max_dist is in km.
        """

        for path in paths:

            _, filename = os.path.split(path)
            filename, file_ext = os.path.splitext(filename)
            network, station, phase = filename.split(".")
            coords = None
            if station_coords is not None:
                coords = station_coords.get(
                    (network, station),
                    station_coords.get(f"{network}.{station}")
                )
            field = fields.read_hdf(path)
            self.add(
                field, "/".join([network, station, phase]),
                station_coords=coords,
                max_dist=max_dist, mask=mask, compress=compress
            )

        return True


    def read(self, key, min_coords=None, max_coords=None):

        group = self.f5[key]

        _coord_sys = group.attrs["coord_sys"]
        _field_type = group.attrs["field_type"]
        _min_coords = group["min_coords"][:]
        _node_intervals = group["node_intervals"][:]
        _npts = group["npts"][:]

        if min_coords is not None:
            min_coords = np.array(min_coords)

        if max_coords is not None:
            max_coords = np.array(max_coords)

        if min_coords is not None and max_coords is not None:
            if np.any(min_coords >= max_coords):
                raise ValueError("All values of min_coords must satisfy min_coords < max_coords")

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
            field = fields.ScalarField3D(coord_sys=_coord_sys)
        elif _field_type == "vector":
            field = fields.VectorField3D(coord_sys=_coord_sys)
        else:
            raise ValueError(f"Unrecognized field type: {_field_type}")

        field.min_coords = _min_coords  +  idx_start * _node_intervals
        field.node_intervals = _node_intervals
        field.npts = idx_end - idx_start
        idxs = tuple(slice(idx_start[idx], idx_end[idx]) for idx in range(3))
        field.values = group["values"][idxs]

        return field


def _solve_one_traveltime(args):
    """
    Multiprocessing worker: one FMM solve. Takes/returns only picklable
    plain data (no open HDF5 handles cross the process boundary).
    """
    key, coords, vm_spec, solver_kwargs = args
    from . import solver as _solver

    if isinstance(vm_spec, (str, os.PathLike)):
        vmodel = fields.read_hdf(vm_spec)
        vm_spec = dict(
            coord_sys=vmodel.coord_sys, min_coords=vmodel.min_coords,
            node_intervals=vmodel.node_intervals, npts=vmodel.npts,
            values=vmodel.values
        )

    slv = _solver.PointSourceSolver(coord_sys=vm_spec["coord_sys"])
    slv.velocity.min_coords = vm_spec["min_coords"]
    slv.velocity.node_intervals = vm_spec["node_intervals"]
    slv.velocity.npts = vm_spec["npts"]
    slv.velocity.values = vm_spec["values"]
    if solver_kwargs:
        for attr, value in solver_kwargs.items():
            setattr(slv, attr, value)
    slv.src_loc = np.asarray(coords, dtype=np.float64)
    slv.solve()

    tt = slv.traveltime
    return key, dict(
        coord_sys=tt.coord_sys,
        min_coords=np.asarray(tt.min_coords),
        node_intervals=np.asarray(tt.node_intervals),
        npts=np.asarray(tt.npts),
        values=np.asarray(tt.values)
    )


def ensure_traveltimes(path, requests, velocity_models, max_dist=None,
                       mask=True, compress=True, solver_kwargs=None,
                       nproc=None):
    """
    Cross-process-safe compute-and-store of missing traveltime grids,
    with parallel FMM solves when several stations are missing at once.

    Three phases:
      1. shared lock  : read-only check of which requested keys are
                        missing (fast path returns in ~1 ms if none).
      2. NO lock      : missing grids are solved in a multiprocessing
                        pool (nproc workers, default = min(n_missing,
                        cpu_count)). Readers are never blocked while
                        solves run.
      3. exclusive lock: keys are re-checked and results written. If a
                        concurrent process stored a grid meanwhile, its
                        copy wins and ours is discarded (duplicate solves
                        across racing processes are possible but
                        harmless; duplicate WRITES are not).

    requests : {("NET","STA","PHASE"): station_coords}
    velocity_models : {"P": ScalarField3D or hdf path, "S": ...}
    Returns {"present": [...], "computed": [...], "skipped": [...]}.

    Locking: <path>.lock, exclusive for writes, shared for reads. Hold a
    shared lock on the same file while reading the inventory elsewhere
    (the SeisComP wrapper does this).
    """
    import fcntl
    import multiprocessing

    requests = {tuple(k): v for k, v in requests.items()}
    result = {"present": [], "computed": [], "skipped": []}

    # ---- phase 1: what is missing? (shared lock, read-only) ----
    missing = dict(requests)
    if os.path.exists(path):
        fd = os.open(path + ".lock", os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            try:
                with TraveltimeInventory(path, mode="r") as inv:
                    for key in requests:
                        if inv.has(key):
                            result["present"].append(key)
                            missing.pop(key)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    if not missing:
        return result

    # resolve velocity model specs; pass file paths through untouched so
    # workers read them independently (cheaper than pickling big arrays)
    vm_specs = {}
    for phase, vm in velocity_models.items():
        if isinstance(vm, (str, os.PathLike)):
            vm_specs[phase.upper()] = str(vm)
        else:
            vm_specs[phase.upper()] = dict(
                coord_sys=vm.coord_sys, min_coords=np.asarray(vm.min_coords),
                node_intervals=np.asarray(vm.node_intervals),
                npts=np.asarray(vm.npts), values=np.asarray(vm.values)
            )

    tasks = []
    for key, coords in missing.items():
        phase = key[2].upper() if len(key) > 2 else "P"
        if phase not in vm_specs or coords is None:
            warnings.warn(
                f"{'/'.join(key)}: cannot compute traveltimes "
                f"({'no velocity model for phase ' + phase if phase not in vm_specs else 'no station coordinates'}); "
                f"skipping.",
                stacklevel=2
            )
            result["skipped"].append(key)
            continue
        tasks.append((key, np.asarray(coords, dtype=np.float64),
                      vm_specs[phase], solver_kwargs))

    if not tasks:
        return result

    # ---- phase 2: parallel solves, no lock held ----
    if nproc is None:
        nproc = min(len(tasks), os.cpu_count() or 1)
    nproc = max(1, int(nproc))
    if nproc == 1 or len(tasks) == 1:
        solved = [_solve_one_traveltime(t) for t in tasks]
    else:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=nproc) as pool:
            solved = pool.map(_solve_one_traveltime, tasks)

    # ---- phase 3: write under exclusive lock, re-checking ----
    coords_by_key = {key: coords for key, coords, _, _ in tasks}
    fd = os.open(path + ".lock", os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        mode = "a" if os.path.exists(path) else "w"
        try:
            with TraveltimeInventory(path, mode=mode) as inv:
                for key, data in solved:
                    if inv.has(key):     # a racing process wrote it first
                        result["present"].append(key)
                        continue
                    field = fields.ScalarField3D(coord_sys=data["coord_sys"])
                    field.min_coords = data["min_coords"]
                    field.node_intervals = data["node_intervals"]
                    field.npts = data["npts"]
                    field.values = data["values"]
                    inv.add(
                        field, "/".join(key),
                        station_coords=coords_by_key[key],
                        max_dist=max_dist, mask=mask, compress=compress
                    )
                    result["computed"].append(key)
                inv.f5.flush()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)

    return result

