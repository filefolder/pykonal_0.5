#!/usr/bin/env python3
"""
pykonal_locext.py — SeisComP LocExt wrapper for the pykonal_0.5 EDT locator.

SeisComP's ExternalLocator (plugin: locext) calls this script with a
SeisComP XML EventParameters document on stdin (one origin + its picks)
and expects a SeisComP XML document containing a single relocated Origin
on stdout.

Configure in global.cfg (or scolv.cfg):

    plugins = ${plugins}, locext
    ExternalLocator.profiles = pykonalEDT:"/path/to/python /path/to/pykonal_locext.py --config /path/to/pykonal_locext.json"

LocExt may additionally pass:
    --max-dist=X                cut-off distance (degrees) — applied if
                                station coordinates are configured
    --fixed-depth=X             fix depth to X km
    --ignore-initial-location   start search from grid center instead of
                                the input origin

All diagnostics go to stderr (visible in scolv's process manager / logs);
stdout carries ONLY the result XML.
"""

import argparse
import contextlib
import fcntl
import json
import math
import os
import sys
import tempfile
from statistics import NormalDist
import traceback

import numpy as np

_REAL_STDOUT = sys.stdout  # replaced with the true fd-1 stream inside main()

import seiscomp.core
import seiscomp.datamodel
import seiscomp.io

from pykonal.locate import EQLocator
from pykonal.transformations import geo2sph, sph2geo
from pykonal import constants as pk_constants

# Mean Earth radius (km) for angular <-> arc-length conversions on the
# spherical grid. Grids are always spherical (pykonal geo2sph); the former
# cartesian / local-projection path has been removed.
MEAN_RADIUS = 6371.0088


def gc_distance_azimuth(lat1, lon1, lat2, lon2):
    """Great-circle distance (degrees of arc) and azimuth 1->2 (degrees)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2)
    dist = math.degrees(2 * math.asin(min(1.0, math.sqrt(a))))
    az = math.degrees(math.atan2(
        math.sin(dlon) * math.cos(p2),
        math.cos(p1) * math.sin(p2)
        - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    )) % 360.0
    return dist, az


def chi_scale(percent, dof):
    """Number of sigmas enclosing `percent` of a `dof`-dimensional Gaussian.

    dof=3 for the confidence ellipsoid, 2 for the horizontal ellipse, 1 for
    a single marginal (latitude, longitude, depth). These differ a lot:
    at 68% they are 1.878, 1.515 and 1.000 respectively, so quoting a
    1-sigma semi-axis as a "68% ellipsoid" understates it by ~47%.

    Exact for dof 1 and 2; Wilson-Hilferty for dof 3 (error < 0.5% over the
    range of levels anyone reports). No scipy dependency.
    """
    p = min(max(percent / 100.0, 1e-6), 1.0 - 1e-9)
    nd = NormalDist()
    if dof == 1:
        return nd.inv_cdf(0.5 * (1.0 + p))
    if dof == 2:
        return math.sqrt(-2.0 * math.log(1.0 - p))
    z = nd.inv_cdf(p)
    return math.sqrt(
        dof * (1.0 - 2.0 / (9.0 * dof) + z * math.sqrt(2.0 / (9.0 * dof))) ** 3
    )


def log(msg):
    print(f"[pykonal_locext] {msg}", file=sys.stderr, flush=True)


def _set_enum(setter, *names):
    """
    Set a seiscomp enum-valued field robustly across versions. For each
    candidate attribute name on seiscomp.datamodel that exists, try the
    setter; keep the first value the setter accepts. A name may resolve to
    a constant from a DIFFERENT enum, which the setter rejects (e.g.
    "enum value out of range") — those are skipped. Returns True on
    success, False if no candidate worked.
    """
    for name in names:
        value = getattr(seiscomp.datamodel, name, None)
        if value is None:
            continue
        try:
            setter(value)
            return True
        except Exception:
            continue
    return False


@contextlib.contextmanager
def inventory_lock(inventory_path, shared):
    """
    Coordinate HDF5 access across concurrent scolv/screloc invocations:
    a shared lock while reading (locating), an exclusive lock while
    ensure_traveltimes() writes new grids. Uses <inventory>.lock, the
    same file TraveltimeInventory.ensure_traveltimes() locks on.
    """
    with open(inventory_path + ".lock", "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


# --------------------------------------------------------------------------
# coordinate transforms between geographic (lat, lon, depth_km) and grid
# --------------------------------------------------------------------------
class GeoTransform:
    """
    Geographic (lat, lon, depth_km) <-> pykonal spherical grid coordinates
    (r, colatitude, longitude) in radians, r = EARTH_RADIUS - depth. Matches
    grids built with pykonal.transformations.geo2sph.

    This deployment is spherical-only. The former cartesian / local-map
    projection path (azimuthal-equidistant about a reference point, with a
    WGS84 radii-of-curvature fallback) has been removed: grids are always
    fully-fledged geographic projections, so geo2sph / sph2geo are the
    correct and only mapping. coord_sys is retained as an attribute (always
    "spherical") so call sites and log labels that key on it keep working.
    """

    def __init__(self, cfg):
        self.coord_sys = cfg.get("coord_sys", "spherical")
        if self.coord_sys != "spherical":
            log(f"WARNING: coord_sys={self.coord_sys!r} is not supported; "
                f"this build is spherical-only (the cartesian path was "
                f"removed). Proceeding as spherical.")
            self.coord_sys = "spherical"

    def geo_to_grid(self, lat, lon, depth_km):
        return geo2sph(np.array([lat, lon, depth_km]))

    def grid_to_geo(self, coords):
        lat, lon, depth = sph2geo_scalar(coords)
        return lat, lon, depth

    def delta_to_grid(self, dlat_km, dlon_km, ddep_km):
        """Search-box half-widths in grid units: depth stays km on the
        radial axis, lateral km -> radians on the sphere."""
        r = pk_constants.EARTH_RADIUS
        return np.array([ddep_km, dlat_km / r, dlon_km / r])


def sph2geo_scalar(coords):
    geo = sph2geo(np.atleast_2d(np.asarray(coords, dtype=float)))[0]
    return float(geo[0]), float(geo[1]), float(geo[2])


# --------------------------------------------------------------------------
# XML I/O helpers
# --------------------------------------------------------------------------
# ---------------------------------------------------------------- sharding
def _shard_path(dirpath, key):
    """Per-station-phase file for key (net, sta, phase)."""
    return os.path.join(dirpath, "{}.{}.{}.h5".format(*key))


def clamp_to_grid(coords, lo, hi, vertical_axis):
    """
    Fit a station onto the velocity grid.

    The VERTICAL coordinate is clamped into range rather than rejected: a
    station a few hundred metres above sea level would otherwise fall
    outside a model whose top is the datum, and dropping it entirely is far
    worse than placing it at the grid's top node (~0.06 s of P traveltime
    for 340 m). Being outside LATERALLY is a real exclusion - there is no
    sensible place to put such a source - so that returns None.

    Returns clamped coordinates, or None if laterally outside.
    """
    c = np.asarray(coords, dtype=float).copy()
    lateral = [i for i in range(3) if i != vertical_axis]
    if any(c[i] < lo[i] or c[i] > hi[i] for i in lateral):
        return None
    c[vertical_axis] = min(max(c[vertical_axis], lo[vertical_axis]),
                           hi[vertical_axis])
    return c


def auto_search_delta(locator, initial_xyz, arrival_keys, sigma_s,
                      coord_sys, k_reach=3.5, cond_max=1.0e6):
    """
    Geometry-based search half-widths from the predicted location
    covariance C = sigma^2 (G^T G)^-1. Each row of G is one arrival's
    traveltime spatial gradient at the trial hypocenter (the ray slowness
    vector) plus a unit origin-time column. Validated against the EDT
    locator's empirical scatter (predicted vs observed sigma agree within
    ~1.5x), so k_reach * sigma is a sound box: large enough to contain the
    true hypocenter, tight enough that the fixed sampling budget resolves
    the likelihood peak.

    Spherical grids: axis 0 is r (radius, km; depth = R - r), axes 1-2 are
    colatitude and longitude in RADIANS. The finite-difference step is one
    node interval per axis (a flat step in radians would land off-grid),
    and the angular derivatives are converted to per-km via the local
    metric (d/dtheta / r, d/dphi / (r sin theta)) so all three columns of G
    are d(tt)/d(distance_km). The returned half-widths are converted back
    to grid units for the locator.

    Returns (delta_grid, info). info has: sigma (per-axis 1-sigma in km, in
    grid-axis order), cond (of G^T G), n (rows used), weak (cond>cond_max).
    Returns (None, info) if too few gradients or the inverse fails.
    """
    spherical = (coord_sys == "spherical")
    r0 = float(initial_xyz[0]) if spherical else None
    theta0 = float(initial_xyz[1]) if spherical else None
    rows = []
    for key in arrival_keys:
        field = locator.traveltimes.get(key) if hasattr(
            locator, "traveltimes") else None
        if field is None:
            continue
        node = np.asarray(field.node_intervals, dtype=float)
        g = np.zeros(3)
        ok = True
        for i in range(3):
            h = node[i]
            xp = np.array(initial_xyz, dtype=float); xp[i] += h
            xm = np.array(initial_xyz, dtype=float); xm[i] -= h
            tp = field.value(xp); tm = field.value(xm)
            if not (np.isfinite(tp) and np.isfinite(tm)):
                ok = False
                break
            g[i] = (tp - tm) / (2.0 * h)
        if not ok:
            continue
        if spherical:
            g[1] = g[1] / r0
            g[2] = g[2] / (r0 * max(np.sin(theta0), 1e-6))
        rows.append([g[0], g[1], g[2], 1.0])

    info = {"n": len(rows), "weak": False, "cond": np.inf, "sigma": None}
    if len(rows) < 4:
        info["weak"] = True
        return None, info
    G = np.array(rows)
    GtG = G.T @ G
    try:
        cond = float(np.linalg.cond(GtG))
        C = sigma_s ** 2 * np.linalg.inv(GtG)
    except np.linalg.LinAlgError:
        info["weak"] = True
        return None, info
    sigma_km = np.sqrt(np.clip(np.diag(C)[:3], 0.0, None))
    info["cond"] = cond
    info["sigma"] = sigma_km
    info["weak"] = cond > cond_max
    hw_km = k_reach * sigma_km          # half-widths in km, grid-axis order
    if spherical:
        delta_grid = np.array([
            hw_km[0],
            hw_km[1] / r0,
            hw_km[2] / (r0 * max(np.sin(theta0), 1e-6)),
        ], dtype=float)
    else:
        delta_grid = hw_km
    return delta_grid, info


def _model_fingerprint(vm_path):
    """
    Cheap identity for a velocity model file: size + mtime. Stored on each
    grid so grids built from a superseded model can be detected and
    rebuilt. Traveltime grids are otherwise cached forever, so editing the
    velocity model silently leaves stale grids that look perfectly valid
    but predict the wrong times.
    """
    try:
        st = os.stat(vm_path)
        return f"{st.st_size}:{int(st.st_mtime)}"
    except Exception:
        return "unknown"


def _shard_is_current(dirpath, key, velocity_models):
    """True if the stored grid exists AND was built from the current model."""
    path = _shard_path(dirpath, key)
    if not os.path.exists(path):
        return False
    vm = velocity_models.get(key[2]) or velocity_models.get(key[2].upper())
    if not isinstance(vm, str):
        return True          # cannot fingerprint an in-memory model
    want = _model_fingerprint(vm)
    try:
        import h5py
        with h5py.File(path, "r") as f5:
            got = f5.attrs.get("velocity_model_fingerprint")
        if isinstance(got, bytes):
            got = got.decode()
        return got == want
    except Exception:
        return False


def _build_one_shard(args):
    """
    Worker: build ONE station-phase grid into its own HDF5 file, using
    pykonal's ensure_traveltimes (which handles its own per-file lock).
    Returns (key, error_or_None). Runs with nproc=1 because parallelism
    happens across shards here, not inside.
    """
    key, coords, velocity_models, dirpath, max_dist = args
    try:
        from pykonal.inventory import ensure_traveltimes
        shard = _shard_path(dirpath, key)
        rep = ensure_traveltimes(
            shard, {key: coords}, velocity_models,
            max_dist=max_dist, nproc=1,
        )
        if key in rep.get("skipped", []):
            return key, "skipped by ensure_traveltimes"

        # Sanity-guard the stored field. A traveltime must be finite and
        # non-negative; a point-source field's minimum must also be AT the
        # source. Rare eikonal-solver glitches can leave an isolated -inf
        # (or a small patch of impossibly low times), which then corrupts
        # the objective for any trial hypocenter near it. Replace such
        # nodes with NaN so they are treated as no-data rather than as an
        # attractive minimum, and report it.
        import h5py
        bad = 0
        with h5py.File(shard, "r+") as f5:
            dset = f5["/".join(key)]["values"]
            vals = dset[...]
            impossible = ~np.isfinite(vals) & ~np.isnan(vals)   # +/-inf
            impossible |= np.isfinite(vals) & (vals < 0)        # negative
            bad = int(impossible.sum())
            if bad:
                vals[impossible] = np.nan
                dset[...] = vals
        if bad:
            # repaired, not failed: the grid is still usable
            log(f"{'.'.join(key)}: {bad} impossible traveltime node(s) "
                f"(-inf or negative) masked to NaN — the eikonal solve "
                f"produced a corrupt patch; check the velocity model there")

        # stamp with the model identity so a later model edit invalidates it
        vm = velocity_models.get(key[2]) or velocity_models.get(key[2].upper())
        if isinstance(vm, str):
            with h5py.File(shard, "r+") as f5:
                f5.attrs["velocity_model_fingerprint"] = _model_fingerprint(vm)
        return key, None
    except Exception as err:
        return key, f"{type(err).__name__}: {err}"


def _rebuild_master(dirpath):
    """
    Rebuild the master index: a small HDF5 file whose groups are EXTERNAL
    LINKS to the per-station files. pykonal's TraveltimeInventory and
    EQLocator follow these transparently, so no pykonal change is needed
    (pyvorotomo keeps using single-file inventories untouched).

    Rebuilt wholesale from a directory scan -- it is a few KB, so this is
    fast and avoids incremental corruption. Links are RELATIVE so the
    directory can be moved or copied as a unit.
    """
    import h5py
    master = os.path.join(dirpath, "inventory.h5")
    tmp = master + ".tmp"
    with h5py.File(tmp, "w") as f5:
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith(".h5") or fname == "inventory.h5":
                continue
            parts = fname[:-3].split(".")
            if len(parts) != 3:
                continue
            net, sta, phase = parts
            f5[f"{net}/{sta}/{phase}"] = h5py.ExternalLink(
                fname, f"/{net}/{sta}/{phase}"
            )
    os.replace(tmp, master)   # atomic swap
    return master


def ensure_traveltimes_sharded(dirpath, requests, velocity_models,
                               max_dist=1000.0, nproc=None):
    """
    Plugin-side alternative to pykonal's ensure_traveltimes that stores one
    HDF5 file per station-phase instead of one large shared file.

    Why: this plugin grows its inventory on demand from a live scolv. With
    a single file, every new grid takes an exclusive lock on the whole
    inventory (blocking all other relocations), HDF5 never reclaims space
    when grids are rebuilt after a velocity-model change, and a crash
    mid-write risks the entire archive. Per-station files make writes
    independent, make rebuilds a file delete, and limit any corruption to
    one station.

    Returns the same {"computed", "present", "skipped"} report shape, plus
    the master index path to hand to EQLocator.
    """
    os.makedirs(dirpath, exist_ok=True)
    report = {"computed": [], "present": [], "skipped": []}

    todo = []
    for key, coords in requests.items():
        if coords is None:
            report["skipped"].append(key)
        elif _shard_is_current(dirpath, key, velocity_models):
            report["present"].append(key)
        else:
            # A stale grid must be DELETED before rebuilding: pykonal's
            # ensure_traveltimes sees the key already present in the file
            # and would skip it, silently keeping the old traveltimes.
            stale = _shard_path(dirpath, key)
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                    log(f"{'.'.join(key)}: grid was built from a superseded "
                        f"velocity model; rebuilding")
                except OSError as err:
                    log(f"{'.'.join(key)}: could not remove stale grid ({err})")
            todo.append((key, coords, velocity_models, dirpath, max_dist))

    if todo:
        n = nproc or min(len(todo), os.cpu_count() or 1)
        if n > 1:
            import multiprocessing
            with multiprocessing.Pool(processes=n) as pool:
                results = pool.map(_build_one_shard, todo)
        else:
            results = [_build_one_shard(t) for t in todo]
        for key, err in results:
            if err is None:
                report["computed"].append(key)
            else:
                log(f"{'.'.join(key)}: {err}")
                report["skipped"].append(key)

    master = _rebuild_master(dirpath)
    report["master"] = master
    return report



def read_input():
    """
    Read EventParameters from stdin and extract everything needed into
    plain Python objects.

    Returns
    -------
    origin_info : dict
        latitude, longitude, depth (km, may be None), time (seismic.Time),
        publicID.
    arrivals : list of dict
        one per arrival: network, station, phase, pick_id, time
        (seismic.Time), time_uncertainty (float or None), weight (float),
        used (bool).
    keepalive : the EventParameters object (hold a reference until done).
    """
    data = sys.stdin.buffer.read()
    return read_input_from_bytes(data)


def read_input_from_bytes(data):
    """
    Same as read_input() but parses EventParameters from an in-memory
    bytes object instead of stdin. Used by the batch driver so many events
    can be relocated in one process. See read_input for the segfault note
    on keeping the parent EventParameters alive.
    """
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        f.write(data)
        path = f.name

    ar = seiscomp.io.XMLArchive()
    if not ar.open(path):
        raise RuntimeError("Could not open input XML")
    obj = ar.readObject()
    ar.close()
    try:
        os.unlink(path)
    except OSError:
        pass

    ep = seiscomp.datamodel.EventParameters.Cast(obj)
    if ep is None:
        raise RuntimeError("Input is not an EventParameters document")
    if ep.originCount() < 1:
        raise RuntimeError("Input contains no origin")

    # index picks by publicID for arrival lookup
    picks = {}
    for i in range(ep.pickCount()):
        pick = ep.pick(i)
        picks[pick.publicID()] = pick

    origin = ep.origin(0)

    def _opt(getter):
        """Return getter().value() or None if the quantity is unset."""
        try:
            return getter().value()
        except Exception:
            return None

    origin_info = {
        "publicID": origin.publicID(),
        "latitude": origin.latitude().value(),
        "longitude": origin.longitude().value(),
        "depth": _opt(origin.depth),
        "time": origin.time().value(),
    }

    arrivals = []
    for i in range(origin.arrivalCount()):
        arr = origin.arrival(i)
        pid = arr.pickID()
        pick = picks.get(pid)
        if pick is None:
            continue

        try:
            weight = arr.weight()
        except Exception:
            weight = 1.0

        # phase: prefer the arrival's phase, else the pick's hint
        phase = None
        try:
            phase = arr.phase().code()
        except Exception:
            try:
                phase = pick.phaseHint().code()
            except Exception:
                phase = "P"
        if not phase:
            phase = "P"

        try:
            time_uncertainty = pick.time().uncertainty()
        except Exception:
            time_uncertainty = None

        wf = pick.waveformID()
        arrivals.append({
            "network": wf.networkCode(),
            "station": wf.stationCode(),
            "phase": phase,
            "pick_id": pid,
            "time": pick.time().value(),
            "time_uncertainty": time_uncertainty,
            "weight": weight,
        })

    return origin_info, arrivals, ep


def write_output(origin, wrap_eventparameters=False):
    """
    Write the relocated origin to the REAL stdout as a SeisComP XML
    document containing the Origin DIRECTLY under <seiscomp>, i.e.

        <seiscomp ...>
          <Origin publicID="..."> ... </Origin>
        </seiscomp>

    This is exactly what the LocExt plugin expects ("a SeisComP XML
    document just containing an origin"). NOTE: the origin must NOT be
    wrapped in <EventParameters> — scolv's reader looks for a top-level
    Origin and, if it finds EventParameters instead, rejects the result
    with "no origin in result document". Writing the Origin object itself
    (not an EventParameters) via XMLArchive produces the correct layout.

    Uses _REAL_STDOUT (captured before we redirected fd 1 to stderr in
    main), so nothing any library printed to stdout can corrupt the
    single XML document the plugin reads from our stdout.
    """
    xml_bytes = origin_to_bytes(origin, wrap_eventparameters=wrap_eventparameters)
    _REAL_STDOUT.buffer.write(xml_bytes)
    _REAL_STDOUT.flush()


def origin_to_bytes(origin, wrap_eventparameters=False):
    """Serialize the relocated Origin to SeisComP XML bytes.

    By default writes a BARE <Origin> directly under <seiscomp>, which is
    what scolv's LocExt reader requires. With wrap_eventparameters=True the
    Origin is placed inside an <EventParameters> element, producing a
    self-contained document that scdispatch can merge/associate directly
    (no separate wrapping step). The arrivals still reference existing pick
    publicIDs; the picks themselves are not emitted, so the target database
    must already hold them (true for relocation of an existing catalog)."""
    if wrap_eventparameters:
        ep = seiscomp.datamodel.EventParameters()
        ep.add(origin)
        obj = ep
        expect = b"<EventParameters"
    else:
        obj = origin
        expect = b"<Origin"
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="r+") as f:
        path = f.name
    ar = seiscomp.io.XMLArchive()
    ar.setFormattedOutput(True)
    if not ar.create(path):
        raise RuntimeError("Could not create output XML")
    ar.writeObject(obj)
    ar.close()
    with open(path, "rb") as f:
        xml_bytes = f.read()
    if not xml_bytes.lstrip().startswith(b"<?xml"):
        raise RuntimeError("generated output does not begin with an XML declaration")
    if expect not in xml_bytes:
        raise RuntimeError(f"generated output missing expected element {expect!r}")
    try:
        os.unlink(path)
    except OSError:
        pass
    return xml_bytes


# --------------------------------------------------------------------------
# main locator logic
# --------------------------------------------------------------------------
def main():
    # Capture the real stdout for the final XML, then redirect fd 1 to fd 2
    # so that anything pykonal / h5py / solvers / warnings might print goes
    # to stderr (visible in scolv logs) and can never corrupt the single
    # XML document the locext plugin reads from our stdout.
    global _REAL_STDOUT
    # Redirect fd 1 -> fd 2 exactly once so stray library output can't
    # corrupt the XML on stdout. Guarded so batch drivers that call main()
    # in a loop don't re-dup on every event.
    if not getattr(main, "_stdout_redirected", False):
        _REAL_STDOUT = os.fdopen(os.dup(1), "w")
        os.dup2(2, 1)
        main._stdout_redirected = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-dist", type=float, default=None)
    parser.add_argument("--fixed-depth", type=float, default=None)
    parser.add_argument("--ignore-initial-location", action="store_true")
    parser.add_argument(
        "--wrap-eventparameters", action="store_true",
        help="wrap the output Origin in an <EventParameters> element so the "
             "document can be fed straight to scdispatch (-O merge) for "
             "standalone/batch relocation. The DEFAULT is a bare <Origin>, "
             "which is what scolv's LocExt reader requires; only use this "
             "flag when driving the plugin outside scolv.")
    args, unknown = parser.parse_known_args()
    if unknown:
        log(f"ignoring unknown args: {unknown}")

    with open(args.config) as f:
        cfg = json.load(f)

    transform = GeoTransform(cfg)
    method = cfg.get("method", "edt")
    alpha = float(cfg.get("alpha", 0.01))
    alpha = np.maximum(alpha,1e-5)

    # Per-phase default pick uncertainty (seconds), used ONLY when a pick
    # carries no stated time uncertainty of its own. A phase->floor map:
    # keys may be exact phase names ("Sg") or a single letter ("S") matching
    # any phase starting with it, with exact names taking precedence. S
    # arrivals are typically less precise than P, so raise the S value if
    # desired. Picks that DO carry their own uncertainty always use that.
    default_pick_error = cfg.get("default_pick_error", {"P": 0.1, "S": 0.1})
    if not isinstance(default_pick_error, dict):
        # tolerate a legacy scalar: apply it to all phases
        default_pick_error = {"P": float(default_pick_error),
                              "S": float(default_pick_error)}
    default_pick_error = {str(k): float(v)
                          for k, v in default_pick_error.items()}
    # a single scalar for event-level uses (auto-box floor, pykonal global
    # fallback): the smallest phase floor, so it never over-inflates.
    _default_err_scalar = (min(default_pick_error.values())
                           if default_pick_error else 0.1)

    def _default_err_for_phase(phase):
        if phase in default_pick_error:
            return default_pick_error[phase]             # exact match wins
        if phase and phase[0] in default_pick_error:
            return default_pick_error[phase[0]]          # letter match
        return _default_err_scalar
    delta_km = cfg.get("search_delta_km", [30.0, 30.0, 15.0])
    _auto_delta = isinstance(delta_km, str) and delta_km.lower() == "auto"
    _auto_k = float(cfg.get("auto_search_k", 6.0))
    _auto_cond_max = float(cfg.get("auto_search_cond_max", 1.0e6))
    if _auto_delta:
        delta_km = [50.0, 50.0, 25.0]   # placeholder; recomputed per event
    delta_t = float(cfg.get("search_delta_t", 10.0))
    nsamples = int(cfg.get("posterior_nsamples", 2048))
    # Number of adaptive importance-sampling rounds inside sample_posterior.
    # The solver refines its proposal to the weighted posterior moments each
    # round (2 is the library default and is plenty for a smooth ellipsoid);
    # raise for very skewed posteriors, lower to 1 for speed. This is the
    # solver's OWN adaptation -- the wrapper no longer resizes the box.
    posterior_rounds = int(cfg.get("posterior_rounds", 2))
    # Fixed seed for posterior sampling so repeated relocations of the
    # same event give IDENTICAL uncertainties (the DE search is already
    # seeded separately). Set posterior_seed to null in the config for
    # independent draws each time (e.g. to gauge sampling variability).
    _pseed = cfg.get("posterior_seed", 8675)
    posterior_seed = None if _pseed is None else int(_pseed)
    # Confidence level (percent) for the reported ellipsoid and
    # uncertainties. 68 matches NonLinLoc, whose confidence ellipsoid is
    # the 68% THREE-DIMENSIONAL region -- 1.878 sigma, not 1 sigma. Earlier
    # versions of this plugin emitted bare 1-sigma semi-axes while calling
    # them a confidence ellipsoid, which made pykonal ellipsoids look about
    # half the linear size of NLL ellipsoids for an identical posterior.
    confidence_level = float(cfg.get("confidence_level", 68.0))

    stations = {}
    if cfg.get("stations_csv"):
        # CSV: NET,STA,lat,lon,elev_m — used for arrival distance/azimuth
        # and the --max-dist filter. Optional but recommended.
        with open(cfg["stations_csv"]) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    net, sta, lat, lon, elev = line.split(",")[:5]
                    if net.lower() == "network": continue
                    stations[(net, sta)] = (float(lat), float(lon), float(elev))
                except Exception as e:
                    print(f"could not parse station line: {line} \n {e}")

    # origin_info: plain dict; arrivals_in: list of plain dicts; ep is a
    # keepalive reference (see read_input) — hold it until main returns.
    origin_info, arrivals_in, _ep_keepalive = read_input()

    # ---------------- initial location (search-box center)
    #
    # The DE search volume is (center +/- search_delta_km). To make
    # successive scolv relocations IDEMPOTENT, the center must not depend
    # on the origin scolv feeds back (it re-seeds each relocate with the
    # previous result; on a flat/multi-modal surface that walks the
    # solution around). By default we therefore anchor the search on a
    # stable, pick-derived reference:
    #   search_anchor = "stations" (default): epicenter at the station
    #       centroid, depth at default_depth.
    #   search_anchor = "grid_center": use cfg["grid_center"].
    #   search_anchor = "origin": legacy behavior, center on the input
    #       origin (NOT idempotent under scolv's feed-back; use only with
    #       a wide search_delta_km).
    # --ignore-initial-location or a configured grid_center force a fixed
    # anchor regardless.
    search_anchor = cfg.get("search_anchor", "origin")
    if args.ignore_initial_location and "grid_center" in cfg:
        search_anchor = "grid_center"

    if search_anchor == "grid_center" and "grid_center" in cfg:
        lat0, lon0, dep0 = cfg["grid_center"]
    elif search_anchor == "stations" and stations:
        # centroid (mean lat/lon) of the stations actually contributing
        # arrivals — a stable, pick-derived anchor independent of the
        # origin scolv feeds back
        _used = {(a["network"], a["station"]) for a in arrivals_in}
        _lats = [stations[k][0] for k in _used if k in stations]
        _lons = [stations[k][1] for k in _used if k in stations]
        if _lats:
            lat0 = float(np.mean(_lats))
            lon0 = float(np.mean(_lons))
        else:
            lat0 = origin_info["latitude"]
            lon0 = origin_info["longitude"]
        dep0 = float(cfg.get("default_depth", 10.0))
    else:
        # "origin" (legacy) or no stations available for a centroid
        lat0 = origin_info["latitude"]
        lon0 = origin_info["longitude"]
        dep0 = origin_info["depth"]
        if dep0 is None:
            dep0 = float(cfg.get("default_depth", 10.0))

    if args.fixed_depth is not None:
        dep0 = args.fixed_depth

    # Diagnostic: fingerprint the INPUT to this call. If two successive
    # relocations differ, comparing these lines shows whether scolv fed
    # back a changed initial origin (different dep0/lat0/lon0) or the same
    # input (=> any output difference is the locator, not the input).
    import hashlib as _hashlib
    _afp = _hashlib.md5(
        repr(sorted(
            (a["network"], a["station"], a["phase"], a["pick_id"])
            for a in arrivals_in
        )).encode()
    ).hexdigest()[:8]
    # Event boundary marker. A run interleaves many events in one stderr
    # stream, so this gives a single unambiguous line to search or split on
    # without altering the content lines below it.
    log("=" * 60)
    log(f"input: initial=({lat0:.4f}, {lon0:.4f}, dep {dep0:.1f} km), "
        f"{len(arrivals_in)} arrivals, fingerprint={_afp}, "
        f"ignore_initial={args.ignore_initial_location}, "
        f"fixed_depth={args.fixed_depth}")

    # ---------------- collect arrivals
    # Reference epoch: earliest pick time, so locator times stay small.
    pick_list = []
    for a in arrivals_in:
        if a["weight"] == 0.0:
            continue  # operator disabled this arrival in scolv
        phase = "S" if str(a["phase"]).upper().startswith("S") else "P"
        key = (a["network"], a["station"], phase)
        pick_list.append(
            (key, a["pick_id"], a["time"], a["time_uncertainty"], a["weight"])
        )

    # Distance census (always on): make it explicit how far each arrival's
    # station is and flag any beyond max_dist_km. Beyond that limit no
    # traveltime grid is built, so the arrival is dropped later with no
    # obvious reason; logging it here removes the mystery.
    _maxd = float(cfg.get("max_dist_km", 1000.0))
    if stations:
        _too_far = []
        _dists = []
        for key, pid, t, terr, wt in pick_list:
            sta = stations.get(key[:2])
            if sta is None:
                _dists.append((key, None))
                continue
            d_deg, _ = gc_distance_azimuth(lat0, lon0, sta[0], sta[1])
            d_km = d_deg * 111.195
            _dists.append((key, d_km))
            if d_km > _maxd:
                _too_far.append((key, d_km))
        _no_coord = [k for k, d in _dists if d is None]
        if _too_far:
            log(f"{len(_too_far)} arrival(s) beyond max_dist_km={_maxd:.0f} km "
                f"(no traveltime grid -> will be excluded): "
                + ", ".join(f"{'.'.join(k)}@{d:.0f}km" for k, d in _too_far))
        if _no_coord:
            log(f"{len(_no_coord)} arrival(s) with no station coordinates in "
                f"stations_csv (distance unknown): "
                + ", ".join('.'.join(k) for k in _no_coord))

    if len(pick_list) < 4:
        raise RuntimeError(f"Only {len(pick_list)} usable picks; need >= 4")


    # --max-dist filter (degrees), if station coordinates are available
    if args.max_dist is not None and stations:
        kept = []
        for key, pid, t, terr, wt in pick_list:
            sta = stations.get(key[:2])
            if sta is None:
                kept.append((key, pid, t, terr, wt))
                continue
            d, _ = gc_distance_azimuth(lat0, lon0, sta[0], sta[1])
            if d <= args.max_dist:
                kept.append((key, pid, t, terr, wt))
        log(f"--max-dist={args.max_dist}: kept {len(kept)}/{len(pick_list)} picks")
        pick_list = kept

    epoch = min(t for _, _, t, _, _ in pick_list)
    arrivals = {}
    pick_errors = {}
    pick_ids = {}
    arrival_weights = {}   # scolv weight per key, echoed back on output
    _n_dup = 0
    for key, pid, t, terr, wt in pick_list:
        if key in arrivals:
            _n_dup += 1
            log(f"duplicate arrival for {'.'.join(key)}; keeping first "
                f"(same network/station/phase key)")
            continue
        arrivals[key] = (t - epoch).length()  # TimeSpan -> seconds
        # Pick uncertainty comes from the pick's own time uncertainty, or
        # default_pick_error. Input arrival weights are deliberately NOT
        # folded in: they arrive as a mixture of unrelated quantities
        # (SeisComP 1/2^index quality classes such as 0.125, and the
        # previous locator's internal weights which can exceed 1), so
        # there is no consistent scale to map onto an uncertainty. Doing so
        # silently deleted low-weight picks and sharpened others past their
        # stated error, flattening the EDT surface and destabilising the
        # solution. Weight 0 still excludes a pick (handled above), and the
        # input weight is echoed back on output unchanged.
        pick_errors[key] = float(terr) if terr else _default_err_for_phase(key[2])
        pick_ids[key] = pid
        arrival_weights[key] = float(wt) if wt else 1.0

    # Reconcile the counts so it is never a mystery where arrivals went
    # between input and location: input -> (weight-0 excluded) -> distance
    # census -> (duplicate-key collapsed) -> unique keys located. Any
    # further reduction below is the off-grid filter and outlier rejection,
    # which log separately.
    _n_zero = sum(1 for a in arrivals_in if a["weight"] == 0.0)
    log(f"arrival census: {len(arrivals_in)} input, "
        f"{_n_zero} weight-0 excluded, {_n_dup} duplicate-key collapsed, "
        f"{len(arrivals)} unique station-phase keys to locate")

    # ---------------- run the locator
    # Optional physical clamp on the depth search box. Off by default:
    # depth_min/depth_max are only applied if set in the config. Leave them
    # unset for models whose grid extends above the datum (negative depth =
    # elevation), where a negative-depth solution is valid. When set, the
    # symmetric search box (initial +/- delta) is recentered and its depth
    # half-width shrunk to fit [depth_min, depth_max].
    depth_min = cfg.get("depth_min")
    depth_max = cfg.get("depth_max")
    if depth_min is not None or depth_max is not None:
        lo_bound = -np.inf if depth_min is None else float(depth_min)
        hi_bound = np.inf if depth_max is None else float(depth_max)
        ddepth = float(delta_km[2])
        dep_lo = max(lo_bound, dep0 - ddepth)
        dep_hi = min(hi_bound, dep0 + ddepth)
        if dep_hi <= dep_lo:  # dep0 outside the allowed range
            dep0 = min(max(dep0, lo_bound), hi_bound)
            dep_lo, dep_hi = lo_bound, hi_bound
        dep0 = 0.5 * (dep_lo + dep_hi)
        delta_km = [delta_km[0], delta_km[1], 0.5 * (dep_hi - dep_lo)]

    initial_xyz = transform.geo_to_grid(lat0, lon0, dep0)
    delta_xyz = transform.delta_to_grid(*[float(d) for d in delta_km])
    if args.fixed_depth is not None:
        # pin depth by collapsing the search interval on the depth axis
        iz = 0 if transform.coord_sys == "spherical" else 2
        delta_xyz[iz] = 1e-3

    initial = np.append(initial_xyz, 0.0)
    delta = np.append(delta_xyz, delta_t)

    grid_stations = {}
    for (net, sta), (slat, slon, selev) in stations.items():
        grid_stations[(net, sta)] = transform.geo_to_grid(slat, slon, -selev / 1000.0)

    # ------------- build / extend the traveltime inventory on demand -------
    # Precomputed grids are canonical; before locating, any station/phase
    # in the pick set that lacks a stored grid is solved once (from the
    # configured velocity_models) and persisted, so this and every later
    # run find it. This also BUILDS the inventory from nothing on the very
    # first event if the file does not yet exist. Serialized across
    # processes via <inventory>.lock. Requires "velocity_models" in the
    # config AND station coordinates (stations_csv); without either, grids
    # cannot be computed and their arrivals are simply excluded.
    # Storage mode. If traveltime_dir is configured, grids are stored one
    # file per station-phase in that directory, with a small master index
    # of HDF5 external links that pykonal reads transparently (no pykonal
    # change; pyvorotomo's single-file inventories are unaffected).
    # Otherwise the legacy single traveltime_inventory file is used.
    shard_dir = cfg.get("traveltime_dir")
    sharded = bool(shard_dir)
    inventory_path = (os.path.join(shard_dir, "inventory.h5")
                      if sharded else cfg.get("traveltime_inventory"))

    velocity_models = cfg.get("velocity_models", {})
    if velocity_models:
        import time as _time
        from pykonal.inventory import (
            ensure_traveltimes, TraveltimeInventory
        )
        requests = {
            (net, sta, phase): grid_stations.get((net, sta))
            for (net, sta, phase) in arrivals
        }
        missing_coords = sorted(
            f"{net}.{sta}" for (net, sta, ph), c in requests.items()
            if c is None
        )
        if missing_coords:
            log(f"no station coordinates for {len(missing_coords)} "
                f"station(s): {', '.join(missing_coords)}; their grids "
                f"cannot be built (add them to stations_csv)")

        # Drop stations whose coordinates fall OUTSIDE the velocity grid
        # BEFORE calling ensure_traveltimes, so pykonal is never asked to
        # solve an out-of-grid source (which would raise and abort the
        # whole location). This keeps the fix entirely in the plugin and
        # leaves shared pykonal behavior untouched. The affected arrivals
        # are excluded and reported; the location proceeds without them.
        _grid_extents = {}   # phase -> (min_coords, max_coords) arrays
        for _ph, _vm in velocity_models.items():
            try:
                from pykonal import fields as _pk_fields
                _f = _pk_fields.read_hdf(_vm) if isinstance(_vm, str) else _vm
                _minc = np.asarray(_f.min_coords, dtype=np.float64)
                _npts = np.asarray(_f.npts)
                _node = np.asarray(_f.node_intervals, dtype=np.float64)
                _grid_extents[_ph.upper()] = (
                    _minc, _minc + (_npts - 1) * _node
                )
            except Exception as _err:
                log(f"could not read velocity model extent for phase "
                    f"{_ph} ({_err}); off-grid pre-filter skipped for it")

        _offgrid = []
        _vax = 0   # spherical: radial (depth) axis is index 0
        for _key in list(requests):
            _c = requests[_key]
            if _c is None:
                continue
            _ext = _grid_extents.get(_key[2].upper())
            if _ext is None:
                continue
            _lo, _hi = _ext
            _fitted = clamp_to_grid(_c, _lo, _hi, _vax)
            if _fitted is None:
                _offgrid.append(_key)      # laterally outside: no grid possible
                del requests[_key]
            else:
                requests[_key] = _fitted   # elevation clamped onto the grid top
        if _offgrid:
            log(f"{len(_offgrid)} station/phase(s) fall outside the "
                f"velocity grid and will be excluded: "
                f"{', '.join('.'.join(k) for k in _offgrid)}")

        # Announce BEFORE solving so the operator knows why scolv is
        # pausing: check (read-only) which requested grids are absent and
        # can actually be built, and print the count/names up front.
        to_build = []
        inv_path = inventory_path
        if sharded:
            # per-station files: presence is a cheap filesystem check, no
            # lock and no need to open the inventory at all
            to_build = [
                k for k, c in requests.items()
                if c is not None
                and not _shard_is_current(shard_dir, k, velocity_models)
            ]
            if not os.path.isdir(shard_dir):
                log(f"traveltime directory {shard_dir} does not exist yet; "
                    f"creating it")
        elif os.path.exists(inv_path):
            try:
                with inventory_lock(inv_path, shared=True), \
                     TraveltimeInventory(inv_path, mode="r") as _inv:
                    to_build = [
                        k for k, c in requests.items()
                        if c is not None and not _inv.has(k)
                    ]
            except Exception as _err:
                log(f"could not pre-scan inventory ({_err}); proceeding")
        else:
            to_build = [k for k, c in requests.items() if c is not None]
            log(f"traveltime inventory {inv_path} does not exist yet; "
                f"building it now")

        _t_build = None
        if to_build:
            # Effective worker count, mirroring ensure_traveltimes():
            # explicit ensure_nproc if set, else min(n_missing, cpu_count).
            _cfg_nproc = cfg.get("ensure_nproc")
            if _cfg_nproc:
                _nproc = max(1, int(_cfg_nproc))
            else:
                _nproc = min(len(to_build), os.cpu_count() or 1)
            log(f"computing {len(to_build)} new traveltime grid(s) before "
                f"locating on {_nproc} CPU(s) "
                f"(one FMM solve each; this can take a while) — "
                f"{', '.join('.'.join(k) for k in to_build)}")
            _t_build = _time.time()

        if sharded:
            report = ensure_traveltimes_sharded(
                shard_dir, requests, velocity_models,
                max_dist=float(cfg.get("max_dist_km", 1000.0)),
                nproc=cfg.get("ensure_nproc"),
            )
        else:
            report = ensure_traveltimes(
                inv_path, requests, velocity_models,
                max_dist=float(cfg.get("max_dist_km", 1000.0)),
                nproc=cfg.get("ensure_nproc"),
            )
        if report["computed"]:
            _elapsed = (
                f" in {_time.time() - _t_build:.1f} s"
                if _t_build is not None else ""
            )
            log(f"computed and stored {len(report['computed'])} new "
                f"traveltime grid(s){_elapsed}: "
                f"{', '.join('.'.join(k) for k in report['computed'])}")
        if report["skipped"]:
            log(f"could not compute grids for "
                f"{', '.join('.'.join(k) for k in report['skipped'])} "
                f"(missing coords or per-phase velocity model); their "
                f"arrivals will be excluded")
    elif not os.path.exists(inventory_path):
        raise RuntimeError(
            f"traveltime inventory {cfg['traveltime_inventory']} does not "
            f"exist and no 'velocity_models' are configured to build it. "
            f"Add a velocity_models block (and stations_csv) to the config, "
            f"or precompute the inventory."
        )
    # -----------------------------------------------------------------------

    # Shared lock in both modes: single-file writes take an exclusive lock
    # elsewhere, and the sharded master is swapped atomically (os.replace),
    # so readers never see a partial index.
    with inventory_lock(inventory_path, shared=True), \
         EQLocator(inventory_path,
                   coord_sys=transform.coord_sys) as locator:
        locator.default_pick_error = _default_err_scalar
        # EDT_OT_WT (NonLinLoc LOCMETH EDT_OT_WT): penalise the pdf by the
        # spread of the per-arrival origin-time estimates. On by default.
        if hasattr(locator, "edt_ot_wt"):
            locator.edt_ot_wt = bool(cfg.get("edt_ot_wt", True))
        # Exponent on the EDT stack; null/absent = arrival count, as in NLL.
        if hasattr(locator, "edt_exponent") and cfg.get("edt_exponent") is not None:
            locator.edt_exponent = float(cfg["edt_exponent"])
        if "edt_reg" in cfg:
            log("NOTE: edt_reg has been removed; it is ignored. EDT_OT_WT "
                "(edt_ot_wt) does the same job without the outlier "
                "sensitivity. You can delete the key from your config.")
        # Optionally solve using only the N geographically-closest arrivals,
        # while still reporting residuals for ALL of them. Distant arrivals
        # in a 1D model carry systematic Pn/model error that can drag the
        # solution; restricting the SOLVE to nearby stations (which sample
        # the crust the model actually represents) can give a cleaner
        # hypocenter, and the far arrivals still get residuals for QC.
        # Ranking is by epicentral distance from the initial origin, using
        # station coordinates from stations_csv; arrivals without coords
        # cannot be ranked and are always placed in the far (report-only)
        # set. 0 or unset = use all arrivals to solve (original behavior).
        _n_closest = int(cfg.get("solve_n_closest", 0))
        _solve_arrivals = arrivals
        _solve_pick_errors = pick_errors
        if _n_closest > 0 and len(arrivals) > _n_closest:
            _ranked = []
            _unranked = []
            for key in arrivals:
                sta = stations.get(key[:2])
                if sta is None:
                    _unranked.append(key)
                    continue
                d_deg, _ = gc_distance_azimuth(lat0, lon0, sta[0], sta[1])
                _ranked.append((d_deg, key))
            _ranked.sort(key=lambda t: t[0])
            _solve_keys = set(k for _, k in _ranked[:_n_closest])
            _solve_arrivals = {k: v for k, v in arrivals.items()
                               if k in _solve_keys}
            _solve_pick_errors = {k: v for k, v in pick_errors.items()
                                  if k in _solve_keys}
            _far = len(arrivals) - len(_solve_arrivals)
            log(f"solve_n_closest={_n_closest}: solving with the "
                f"{len(_solve_arrivals)} closest arrival(s); {_far} farther "
                f"arrival(s) kept for residual reporting only"
                + (f" ({len(_unranked)} unrankable, no coords)"
                   if _unranked else ""))

        locator.add_arrivals(_solve_arrivals)
        locator.add_pick_errors(_solve_pick_errors)
        if grid_stations:
            locator.add_stations(grid_stations)

        # ----- automatic, geometry-based search box (search_delta_km: auto)
        _auto_delta_unclamped = None   # set to the unclamped depth box below
        if _auto_delta:
            # Two-pass auto sizing. The search box must scale with the
            # ACTUAL location uncertainty, which is set by how well the
            # model fits -- the a posteriori data variance s^2 = sum(r^2) /
            # (N - M) -- NOT by the nominal pick error, which is only a
            # floor. We cannot know the misfit before locating, so:
            #   pass 1: locate in a wide provisional box, measure the rms
            #           residual over the solve set;
            #   pass 2: size the real box from s^2 (G^T G)^-1 and relocate.
            # For an all-Pn regional event this correctly turns a spuriously
            # tiny (tens of metres) box into a realistic few-km box.
            _prov = transform.delta_to_grid(150.0, 150.0, 60.0)
            _prov = np.minimum(
                _prov, transform.delta_to_grid(
                    *np.array(cfg.get("auto_search_max_km",
                                      [200.0, 200.0, 60.0]), dtype=float)))
            _prov_full = np.append(_prov, delta_t)
            locator.read_traveltimes(
                min_coords=(initial_xyz - _prov),
                max_coords=(initial_xyz + _prov))
            _prov_soln = locator.locate(initial, _prov_full, alpha, method)
            _prov_res = locator.residuals(_prov_soln)
            if _prov_res:
                _rv = np.array(list(_prov_res.values()), dtype=float)
                _dof = max(len(_rv) - 4, 1)
                _s_data = float(np.sqrt(np.sum(_rv ** 2) / _dof))
            else:
                _s_data = _default_err_scalar
            # floor at the nominal pick error: a genuinely excellent fit
            # should not size the box below pick precision.
            _s_eff = max(_s_data, _default_err_scalar)

            _adx, _ainfo = auto_search_delta(
                locator, initial_xyz, list(_solve_arrivals.keys()),
                _s_eff, transform.coord_sys,
                k_reach=_auto_k, cond_max=_auto_cond_max)
            if _adx is None:
                _adx = transform.delta_to_grid(100.0, 100.0, 40.0)
                log(f"search_delta auto: geometry too weak to size the box "
                    f"(insufficient/near-singular gradients); wide fallback "
                    f"(s_data={_s_data:.2f}s)")
            else:
                _cap = np.array(cfg.get("auto_search_max_km",
                                        [200.0, 200.0, 60.0]), dtype=float)
                _cap_xyz = transform.delta_to_grid(*_cap)
                _adx = np.minimum(_adx, _cap_xyz)
                _sig = _ainfo["sigma"]           # km, grid-axis order
                _hw = _auto_k * _sig
                if transform.coord_sys == "spherical":
                    _lab = (_sig[1], _sig[2], _sig[0], _hw[1], _hw[2], _hw[0])
                else:
                    _lab = (_sig[0], _sig[1], _sig[2], _hw[0], _hw[1], _hw[2])
                _weak = (" [WEAK geometry: cond=%.1e, depth poorly "
                         "constrained]" % _ainfo["cond"]
                         if _ainfo["weak"] else "")
                log(f"search_delta auto: data_std={_s_data:.2f}s "
                    f"(floor {_default_err_scalar:.2f}s) -> 1-sigma(km) "
                    f"lat={_lab[0]:.2f} lon={_lab[1]:.2f} dep={_lab[2]:.2f}, "
                    f"k={_auto_k:.1f} -> half-widths(km) lat={_lab[3]:.1f} "
                    f"lon={_lab[4]:.1f} dep={_lab[5]:.1f} "
                    f"(cond={_ainfo['cond']:.1e}, n={_ainfo['n']}){_weak}")
            delta = np.append(_adx, delta_t)
            # Preserve the UNCLAMPED auto depth half-width for the posterior:
            # the location box depth is clamped (below) to stop the
            # hypocenter wandering, but the posterior must sample the full
            # depth extent or it truncates the (honestly large) depth
            # uncertainty on unconstrained events.
            _auto_delta_unclamped = np.asarray(delta[:3], dtype=float).copy()

            # Apply the depth clamp to the AUTO box. The early clamp ran on
            # the placeholder delta (auto is not known until here), so the
            # real auto depth half-width has not yet been bounded. Without
            # this, an all-Pn event whose depth is unconstrained gets a
            # large auto depth box and the located depth wanders to the box
            # edge. Recenter depth into [depth_min, depth_max] and shrink
            # the depth half-width to fit, exactly as the fixed-delta path
            # does. Horizontal axes are untouched -- only depth is clamped.
            if depth_min is not None or depth_max is not None:
                _iz = 0 if transform.coord_sys == "spherical" else 2
                _lo = -np.inf if depth_min is None else float(depth_min)
                _hi = np.inf if depth_max is None else float(depth_max)
                # current depth half-width in km
                _dep_hw_km = float(_hw[2]) if _ainfo.get("sigma") is not None \
                    else float(cfg.get("auto_search_max_km",
                                       [200.0, 200.0, 60.0])[2])
                _dlo = max(_lo, dep0 - _dep_hw_km)
                _dhi = min(_hi, dep0 + _dep_hw_km)
                if _dhi <= _dlo:
                    _dc = min(max(dep0, _lo), _hi)
                    _dlo, _dhi = _lo, _hi
                else:
                    _dc = 0.5 * (_dlo + _dhi)
                # rebuild the depth component of initial and delta in grid
                _dep_hw_clamped = 0.5 * (_dhi - _dlo)
                _init_dep_xyz = transform.geo_to_grid(lat0, lon0, _dc)
                initial[_iz] = _init_dep_xyz[_iz]
                _dep_delta_xyz = transform.delta_to_grid(0.0, 0.0, _dep_hw_clamped)
                delta[_iz] = _dep_delta_xyz[_iz]

        _t_compute = _time.time()
        soln = locator.locate(initial, delta, alpha, method)

        # Residuals for ALL arrivals at the final hypocenter, not just the
        # solve set: a residual is observed - predicted at the fixed
        # solution and does not depend on which arrivals drove the fit, so
        # we add the full set back before reading residuals.
        if _solve_arrivals is not arrivals:
            locator.clear_arrivals()
            locator.add_arrivals(arrivals)
            locator.add_pick_errors(pick_errors)
            # residuals() needs the traveltime grids in memory; locate()
            # only loaded the solve-set grids, so re-read for the full set
            # (same search box) or the far arrivals come back with no
            # residual and get nulled in the output.
            locator.read_traveltimes(min_coords=(initial - delta)[:3],
                                     max_coords=(initial + delta)[:3])
        residuals = locator.residuals(soln)

        # ------------------------------------------------ outlier rejection
        # EDT already ignores wildly inconsistent picks when LOCATING, but
        # they still inflate the reported RMS, and a mutually-consistent
        # group of bad picks (e.g. from a neighbouring event) can build a
        # competing peak in the likelihood. So: relocate iteratively,
        # dropping arrivals whose residual is clearly out of family.
        #
        # The cutoff is max(outlier_cutoff_s, outlier_mad_factor * MAD).
        # The MAD term adapts to how noisy this particular event is; the
        # absolute floor stops a very clean event from rejecting picks that
        # are merely a bit worse than its excellent others. Rejected picks
        # are not deleted from the output - they appear with weight 0 and
        # timeUsed false, so the analyst sees which ones were dropped.
        _cut_abs = float(cfg.get("outlier_cutoff_s", 3.0))
        _cut_mad = float(cfg.get("outlier_mad_factor", 4.0))
        _min_keep = max(int(cfg.get("outlier_min_arrivals", 6)), 4)
        # The solve set for relocation is the (possibly restricted) set;
        # residuals are always evaluated over ALL remaining arrivals. When
        # solve_n_closest is active, outlier rejection still uses residuals
        # from every arrival to decide what is bad, but relocates only on
        # the closest set so the restriction is preserved across passes.
        _excluded = []
        if _cut_abs > 0 and len(arrivals) > _min_keep:
            for _pass in range(3):
                _vals = np.array(list(residuals.values()), dtype=float)
                _med = float(np.median(_vals))
                _mad = float(np.median(np.abs(_vals - _med))) * 1.4826
                _cut = max(_cut_abs, _cut_mad * _mad)
                _bad = [k for k, r in residuals.items() if abs(r - _med) > _cut]
                # never drop below the minimum, worst-first
                _bad.sort(key=lambda k: -abs(residuals[k] - _med))
                _bad = _bad[:max(0, len(arrivals) - _min_keep)]
                if not _bad:
                    break
                for k in _bad:
                    arrivals.pop(k, None)
                    pick_errors.pop(k, None)
                    _solve_arrivals.pop(k, None)
                    _solve_pick_errors.pop(k, None)
                _excluded.extend(_bad)
                locator.clear_arrivals()
                locator.add_arrivals(_solve_arrivals)
                locator.add_pick_errors(_solve_pick_errors)
                soln = locator.locate(initial, delta, alpha, method)
                # residuals over ALL remaining arrivals at the new solution
                if _solve_arrivals is not arrivals:
                    locator.clear_arrivals()
                    locator.add_arrivals(arrivals)
                    locator.add_pick_errors(pick_errors)
                    locator.read_traveltimes(min_coords=(initial - delta)[:3],
                                             max_coords=(initial + delta)[:3])
                residuals = locator.residuals(soln)
            if _excluded:
                log(f"rejected {len(_excluded)} outlier arrival(s) "
                    f"(cutoff {_cut:.2f} s): "
                    f"{', '.join('.'.join(k) for k in _excluded)}")

        quality = locator.quality(soln)

        # Grid-axis index mapping, defined ONCE here because it is used both
        # by the box-limited warning during sampling and by the uncertainty
        # reporting further down. It used to be defined only at the second
        # of those, which raised
        #   "cannot access local variable '_ilat'"
        # from the warning path -- i.e. exactly on the events the warning
        # exists to flag.
        # spherical grid axis order (r, theta, phi) = (depth, lat, lon)
        _ilon, _ilat, _iz = 2, 1, 0

        posterior = None
        if method == "edt" and nsamples > 0:
            try:
                # Single generous-box importance sample. The EDT posterior
                # proposal is built from the Hessian (local curvature) of the
                # likelihood at the solution, so the reported uncertainty
                # comes from the DATA, not from the box: the box is only a
                # rejection boundary. An oversized box is therefore harmless
                # -- verified on the solver, sigma is stable and ESS healthy
                # from ~4x sigma out to 40x with no upper failure -- while a
                # box SMALLER than the peak truncates the pdf and understates
                # the uncertainty (the box_limited case). So we do NOT
                # contract: we size the box generously from the a-posteriori
                # geometry prediction (auto_search_delta already returns
                # k * sigma with k = auto_search_k) and sample once. No
                # shifting boxes, no shrink-to-fit; the failure mode is
                # entirely one-sided and we stay on the safe side of it.
                _d = np.asarray(delta[:3], dtype=float).copy()
                # sample the posterior over the UNCLAMPED depth extent so the
                # depth uncertainty is not truncated by the location clamp.
                if _auto_delta_unclamped is not None:
                    _d = _auto_delta_unclamped.copy()
                posterior = locator.sample_posterior(
                    soln[:3], _d, nsamples=nsamples, nscatter=0,
                    seed=posterior_seed, rounds=posterior_rounds,
                    search_delta=np.asarray(delta[:3], dtype=float)
                )
                if posterior.get("box_limited"):
                    # Report the per-axis situation factually and let the
                    # analyst judge. box_fill is sigma / search half-width per
                    # axis; a value above ~0.25 means the box half-width is
                    # under 4 sigma, so that axis's reported uncertainty is a
                    # lower bound (the pdf reaches the box edge). No verdict on
                    # whether that is "good enough" -- just the numbers.
                    try:
                        _bf = np.asarray(posterior.get("box_fill", []), dtype=float)
                        _half = np.abs(np.asarray(_d, dtype=float)) * (
                            posterior.get("metric_scale", np.ones(3))
                        )
                        _axmap = [(_ilat, "north-south"),
                                  (_ilon, "east-west"),
                                  (_iz, "depth")]
                        _parts = []
                        for _g, _lab in _axmap:
                            _s_km = _bf[_g] * _half[_g]
                            _parts.append(
                                f"{_lab} sigma {_s_km:.1f} km / half-width "
                                f"{_half[_g]:.0f} km (fill {_bf[_g]:.2f})")
                        log("posterior reaches the search box edge on one or "
                            "more axes (fill > ~0.25: " + "; ".join(_parts) + ". Widening "
                            "search_delta_km raises a truncated axis; an axis "
                            "the geometry does not constrain (often depth) "
                            "stays wide regardless.")
                    except Exception as _e:
                        log(f"(box-fill diagnostic unavailable: {_e})")
                _corr = posterior.get("depth_time_correlation")
                if _corr is not None and np.isfinite(_corr) and abs(_corr) > 0.9:
                    log(f"depth and origin time are correlated at "
                        f"{_corr:+.2f}: they trade off almost completely, so "
                        f"neither is independently determined. Depth is only "
                        f"as good as the origin time here.")
                _mcse = posterior.get("sigma_rel_mcse")
                if _mcse is not None and np.isfinite(_mcse) and _mcse > 0.15:
                    log(f"sigma has a Monte Carlo standard error of "
                        f"{100*_mcse:.0f}% -- raise posterior_nsamples if you "
                        f"need the uncertainty itself to be precise")
                _prop = str(posterior.get("proposal", "unknown"))
                if "fallback" in _prop:
                    log(f"posterior: curvature-based proposal unusable, fell "
                        f"back to {_prop}; uncertainties are less reliable")
                _ess = float(posterior.get("ess", float("nan")))
                if _ess < 50:
                    log(f"posterior: effective sample size {_ess:.0f} of "
                        f"{nsamples} - uncertainties are poorly resolved; "
                        f"raise posterior_nsamples or narrow search_delta_km")
            except Exception as err:
                log(f"posterior sampling failed: {err}")
                posterior = None

        _compute_seconds = _time.time() - _t_compute

        # traveltimes for per-arrival predicted times were already read
        lat, lon, dep = transform.grid_to_geo(soln[:3])
        t0 = seiscomp.core.Time(epoch) + seiscomp.core.TimeSpan(float(soln[3]))

        # Uncertainties in km from the EDT posterior covariance, at the
        # configured confidence level, computed here so they can be reported
        # inline with the solution and reused below when populating the
        # origin. None if no posterior.
        # The library now reports covariance_km: the posterior covariance
        # already expressed in km^2 in the local orthonormal frame
        # (up, south, east for spherical grids). Prefer
        # it. The fallback path reproduces the old hand-rolled scaling so
        # this script still runs against a pykonal that has not been
        # rebuilt -- but note that only the fallback's per-axis sigmas were
        # ever correct; its ellipsoid was an eigendecomposition of a matrix
        # mixing km with radians.
        lat_km = lon_km = dz = None
        cov_km = None
        if posterior is not None:
            cov_km = posterior.get("covariance_km")
            if cov_km is None:
                _cov = posterior["covariance"]
                _r0 = pk_constants.EARTH_RADIUS - dep
                _s = np.array([1.0, _r0,
                               _r0 * math.sin(math.radians(90.0 - lat))])
                cov_km = np.asarray(_cov) * np.outer(_s, _s)
            cov_km = np.asarray(cov_km, dtype=float)
            # _ilon / _ilat / _iz are set above, before posterior sampling.
            # marginal (1-dof) uncertainties at the requested level
            _k1 = chi_scale(confidence_level, 1)
            lat_km = _k1 * math.sqrt(max(cov_km[_ilat, _ilat], 0.0))
            lon_km = _k1 * math.sqrt(max(cov_km[_ilon, _ilon], 0.0))
            dz     = _k1 * math.sqrt(max(cov_km[_iz, _iz], 0.0))

    if lat_km is not None:
        _ess_txt = ""
        try:
            _ess_txt = f" ess={float(posterior['ess']):.0f}"
        except Exception:
            pass
        _t_txt = ""
        try:
            _ts = float(posterior["time_sigma"])
            if np.isfinite(_ts):
                _t_txt = f", t0 \u00b1{_ts:.2f} s"
        except Exception:
            pass
        _unc = (f"lat {lat:.4f}\u00b0 \u00b1{lat_km:.2f} km, "
                f"lon {lon:.4f}\u00b0 \u00b1{lon_km:.2f} km, "
                f"depth {dep:.1f} \u00b1{dz:.2f} km{_t_txt} "
                f"({confidence_level:.0f}% marginal){_ess_txt}")
    else:
        _unc = (f"lat {lat:.4f}\u00b0, lon {lon:.4f}\u00b0, "
                f"depth {dep:.1f} km (uncertainties not estimated)")
    # Per-arrival detail (opt-in): distance, predicted traveltime and the
    # implied apparent velocity. Because EDT locates on DIFFERENTIAL times,
    # a systematic traveltime error largely cancels in the location and is
    # absorbed by the origin time -- so the hypocenter can match another
    # locator while absolute residuals stay large. Comparing the apparent
    # velocity here against your velocity model is the direct check: if it
    # is consistently higher than the model, the grids were built from a
    # model that is too fast (the usual cause of large positive residuals).
    if cfg.get("log_arrival_detail") and residuals:
        log("per-arrival detail (dist_km, tt_pred_s, apparent_km_s, resid_s):")
        # compute distance for each arrival first so the table can be sorted
        # closest-station-first; arrivals with no coordinates sort last.
        _detail = []
        for key in residuals:
            _sta = stations.get(key[:2])
            if _sta is None:
                continue
            _ddeg, _ = gc_distance_azimuth(lat, lon, _sta[0], _sta[1])
            _detail.append((_ddeg * 111.195, key))
        for _d, key in sorted(_detail, key=lambda t: t[0]):
            # predicted traveltime = observed_offset - t0_offset - residual
            _ttp = arrivals[key] - float(soln[3]) - residuals[key]
            _app = (_d / _ttp) if _ttp > 0.01 else float("nan")
            log(f"    {'.'.join(key):18s} {_d:7.1f} {_ttp:7.2f} "
                f"{_app:7.2f} {residuals[key]:+7.2f}")

    # Residual breakdown: EDT is outlier-robust, so it can produce a good
    # hypocenter while a few bad picks still carry large residuals. Those
    # picks inflate the plain (unweighted) RMS reported above even though
    # the location barely used them. Report the worst offenders and the
    # RMS with them removed, so a high RMS can be attributed to specific
    # picks rather than to a poor location.
    if residuals:
        _r = sorted(residuals.items(), key=lambda kv: -abs(kv[1]))
        _vals = np.array([v for _, v in _r], dtype=float)
        _n_big = int(np.sum(np.abs(_vals) > 1.0))
        _worst = ", ".join(
            f"{'.'.join(k)}={v:+.2f}s" for k, v in _r[:3]
        )
        if len(_vals) > 3:
            _rms_trim = float(np.sqrt(np.mean(np.sort(_vals ** 2)[:-2])))
            log(f"residuals: worst {_worst}; "
                f"{_n_big}/{len(_vals)} exceed 1.0 s; "
                f"rms excluding 2 worst = {_rms_trim:.3f}s")
        else:
            log(f"residuals: worst {_worst}")

    log(f"solution: {_unc} "
        f"rms={quality['rms']:.3f}s nobs={quality['nobs']} "
        f"(computed in {_compute_seconds:.2f} s)")

    # ---------------- build the output Origin
    origin = seiscomp.datamodel.Origin.Create()
    origin.setLatitude(seiscomp.datamodel.RealQuantity(lat))
    origin.setLongitude(seiscomp.datamodel.RealQuantity(lon))
    origin.setDepth(seiscomp.datamodel.RealQuantity(dep))
    # Origin time, with its uncertainty when the posterior supplied one.
    # Previously the time was emitted bare, implying an exactness it never
    # had -- on geometry without a near-source station the origin time
    # trades off almost completely against depth (see the correlation
    # logged above) and can be uncertain by seconds.
    _tq = seiscomp.datamodel.TimeQuantity(t0)
    if posterior is not None:
        _ts = posterior.get("time_sigma")
        if _ts is not None and np.isfinite(_ts):
            try:
                _tq.setUncertainty(float(_ts))
            except Exception:
                pass
    origin.setTime(_tq)
    origin.setMethodID(f"pykonal-{method.upper()}")
    origin.setEarthModelID(cfg.get("earth_model_id", "pykonal"))
    _set_enum(origin.setEvaluationMode,
              "AUTOMATIC", "EvaluationMode_AUTOMATIC")
    if args.fixed_depth is not None:
        _set_enum(origin.setDepthType,
                  "OPERATOR_ASSIGNED", "OriginDepthType_OPERATOR_ASSIGNED")

    ci = seiscomp.datamodel.CreationInfo()
    ci.setCreationTime(seiscomp.core.Time.GMT())
    ci.setAuthor("pykonal_locext")
    origin.setCreationInfo(ci)

    # arrivals with residuals
    #
    # SeisComP validates Arrival.distance on serialization: an Arrival with
    # no distance set throws "Arrival.distance is not set" and aborts the
    # whole output. That happens for any station missing from stations_csv
    # (no coordinates -> no distance). Such arrivals were already excluded
    # from the LOCATION, so we simply do not emit an Arrival for them here;
    # emitting a distance-less one would crash the plugin. Logged so the
    # dropped station is visible.
    used = 0
    _no_dist = []
    for key, pid in pick_ids.items():
        sta = stations.get(key[:2])
        if sta is None:
            _no_dist.append(key)
            continue  # cannot set distance -> would crash serialization
        arr = seiscomp.datamodel.Arrival()
        arr.setPickID(pid)
        arr.setPhase(seiscomp.datamodel.Phase(key[2]))
        if key in residuals:
            arr.setTimeResidual(float(residuals[key]))
            arr.setWeight(arrival_weights.get(key, 1.0))
            arr.setTimeUsed(True)
            used += 1
        else:
            arr.setWeight(0.0)
            arr.setTimeUsed(False)
        dist, az = gc_distance_azimuth(lat, lon, sta[0], sta[1])
        arr.setDistance(dist)  # degrees of arc
        arr.setAzimuth(az)
        origin.add(arr)
    if _no_dist:
        log(f"{len(_no_dist)} arrival(s) omitted from output (no station "
            f"coordinates, cannot set Arrival.distance): "
            + ", ".join('.'.join(k) for k in _no_dist))

    # quality
    q = seiscomp.datamodel.OriginQuality()
    q.setUsedPhaseCount(used)
    q.setAssociatedPhaseCount(len(pick_ids))
    if np.isfinite(quality["rms"]):
        q.setStandardError(float(quality["rms"]))
    if np.isfinite(quality["azimuthal_gap"]):
        q.setAzimuthalGap(float(quality["azimuthal_gap"]))
    if np.isfinite(quality["min_station_dist"]):
        q.setMinimumDistance(math.degrees(
            float(quality["min_station_dist"]) / MEAN_RADIUS))
    origin.setQuality(q)

    # uncertainty ellipsoid from the EDT posterior
    if posterior is not None:
        # ellipsoid_km is the eigendecomposition of the km-frame covariance:
        # physically meaningful semi-axis lengths and genuinely orthonormal
        # principal axes. The legacy "ellipsoid" key is an eigendecomposition
        # of the RAW grid covariance, whose entries on a spherical grid are a
        # mixture of km^2, km*rad and rad^2 -- its semi-axes were not lengths
        # and its eigenvectors were not directions. Fall back to recomputing
        # from cov_km if running against an older library build.
        _ell_km = posterior.get("ellipsoid_km")
        if _ell_km is not None:
            semi_km = np.asarray(_ell_km["semi_axes"], dtype=float)
            axes = np.asarray(_ell_km["axes"], dtype=float)
        else:
            _ev, _evec = np.linalg.eigh(cov_km)
            _order = np.argsort(_ev)[::-1]
            semi_km = np.sqrt(np.clip(_ev[_order], 0.0, None))
            axes = _evec[:, _order].T

        # Scale 1-sigma to the requested confidence level. The ellipsoid is a
        # 3-D region, so the chi-square quantile with 3 degrees of freedom is
        # the right factor (1.878 at 68%), matching NonLinLoc.
        _k3 = chi_scale(confidence_level, 3)
        semi = semi_km * _k3 * 1000.0  # km -> m

        ce = seiscomp.datamodel.ConfidenceEllipsoid()
        ce.setSemiMajorAxisLength(float(semi[0]))
        ce.setSemiIntermediateAxisLength(float(semi[1]))
        ce.setSemiMinorAxisLength(float(semi[2]))
        # spherical axes (r, theta, phi) ~ (up, south, east)
        r, th, ph = axes[0]
        ce.setMajorAxisAzimuth(math.degrees(math.atan2(ph, -th)) % 360.0)
        ce.setMajorAxisPlunge(math.degrees(math.atan2(-r, math.hypot(th, ph))))
        ce.setMajorAxisRotation(0.0)

        ou = seiscomp.datamodel.OriginUncertainty()
        ou.setConfidenceEllipsoid(ce)
        # State the level explicitly. Without it a consumer cannot tell a
        # 1-sigma ellipsoid from a 68% or 95% one, and they differ by
        # factors of 1.88 and 2.80 respectively.
        try:
            ou.setConfidenceLevel(float(confidence_level))
        except Exception:
            pass
        # The enum constant for "confidence ellipsoid" varies by seiscomp
        # version, and a module attribute of the right NAME may belong to a
        # different enum (setter then rejects it as "out of range"). So we
        # try each candidate against the setter and keep the first accepted;
        # if none work, leave preferredDescription unset (the ellipsoid is
        # still attached and used by scolv regardless).
        # Try to set preferredDescription = confidence ellipsoid; on some
        # seiscomp builds no accepted enum value exists, which is fine —
        # the ellipsoid is attached regardless, so we don't warn about it.
        _set_enum(
            ou.setPreferredDescription,
            "CONFIDENCE_ELLIPSOID", "ConfidenceEllipsoid",
            "OU_CONFIDENCE_ELLIPSOID",
            "OriginUncertaintyDescription_CONFIDENCE_ELLIPSOID",
        )
        # Per-coordinate uncertainties in KM (SeisComP's convention for
        # lat/lon/depth uncertainty is kilometers, not degrees). These
        # populate origin.latitude/longitude/depth .uncertainty(), which is
        # what scolv's location view displays. They are MARGINAL (1-dof)
        # intervals at `confidence_level`, so at the 68% default they are
        # very nearly plain 1-sigma; the ellipsoid above is the 3-dof
        # region at the same level and is correspondingly wider. Computed
        # inline with the solution above from covariance_km.
        # horizontalUncertainty (km): the semi-major axis of the HORIZONTAL
        # marginal ellipse at the requested level -- not max(lat, lon). When
        # the horizontal errors are correlated (routine for a one-sided
        # network, where the ellipse is a long diagonal ridge) the true
        # semi-major exceeds both marginal sigmas, sometimes by a lot.
        _ch = cov_km[np.ix_([_ilon, _ilat], [_ilon, _ilat])]
        _hmax = float(np.sqrt(max(np.linalg.eigvalsh(_ch)[-1], 0.0)))
        ou.setHorizontalUncertainty(_hmax * chi_scale(confidence_level, 2))
        origin.setUncertainty(ou)

        # latitude / longitude uncertainties (km) — mirror what we do for
        # depth so scolv shows all three
        _lat = origin.latitude()
        _lat.setUncertainty(float(lat_km))
        origin.setLatitude(_lat)

        _lon = origin.longitude()
        _lon.setUncertainty(float(lon_km))
        origin.setLongitude(_lon)

        # depth uncertainty (km)
        d = origin.depth()
        d.setUncertainty(float(dz))
        origin.setDepth(d)

    write_output(origin, wrap_eventparameters=args.wrap_eventparameters)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)