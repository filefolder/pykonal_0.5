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
import traceback

import numpy as np

_REAL_STDOUT = sys.stdout  # replaced with the true fd-1 stream inside main()

try:
    from geographiclib.geodesic import Geodesic
    _GEODESIC = Geodesic.WGS84
except ImportError:
    _GEODESIC = None

import seiscomp.core
import seiscomp.datamodel
import seiscomp.io

from pykonal.locate import EQLocator
from pykonal.transformations import geo2sph, sph2geo
from pykonal import constants as pk_constants

# --- WGS84 ellipsoid helpers (no flat km-per-degree assumptions) -----------
WGS84_A = 6378.137            # semi-major axis, km
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
MEAN_RADIUS = 6371.0088       # km, for angular distance conversions


def meridian_radius(lat_deg):
    """Meridian radius of curvature M(lat), km: length of 1 radian of latitude."""
    s2 = math.sin(math.radians(lat_deg)) ** 2
    return WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * s2) ** 1.5


def parallel_radius(lat_deg):
    """N(lat)*cos(lat), km: length of 1 radian of longitude at this latitude."""
    lat = math.radians(lat_deg)
    s2 = math.sin(lat) ** 2
    return WGS84_A / math.sqrt(1.0 - WGS84_E2 * s2) * math.cos(lat)


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
    coord_sys = "spherical": pykonal convention (r, colatitude, longitude),
        radians, r = EARTH_RADIUS - depth. Matches grids built with
        pykonal.transformations.geo2sph.
    coord_sys = "cartesian": local grid with x=east km, y=north km,
        z=depth km, relative to (ref_lat, ref_lon) at (x0, y0). Uses the
        same simple-degrees conversion as NLL's TRANS SDC, so grids built
        for an NLL SDC project line up.
    """

    def __init__(self, cfg):
        self.coord_sys = cfg.get("coord_sys", "spherical")
        if self.coord_sys == "cartesian":
            self.ref_lat = float(cfg["ref_lat"])
            self.ref_lon = float(cfg["ref_lon"])
            self.ref_x = float(cfg.get("ref_x", 0.0))
            self.ref_y = float(cfg.get("ref_y", 0.0))

    def geo_to_grid(self, lat, lon, depth_km):
        if self.coord_sys == "spherical":
            return geo2sph(np.array([lat, lon, depth_km]))
        if _GEODESIC is not None:
            # azimuthal equidistant about (ref_lat, ref_lon) via exact WGS84
            # geodesics: range and bearing from the reference are exact at
            # any latitude and offset.
            gd = _GEODESIC.Inverse(self.ref_lat, self.ref_lon, lat, lon)
            s_km = gd["s12"] / 1000.0
            az = math.radians(gd["azi1"])
            x = self.ref_x + s_km * math.sin(az)
            y = self.ref_y + s_km * math.cos(az)
            return np.array([x, y, depth_km])
        # fallback: WGS84 radii-of-curvature equirectangular (sub-km at
        # +/-2 deg offsets; install geographiclib for exact geodesics)
        mid = 0.5 * (lat + self.ref_lat)
        y = self.ref_y + math.radians(lat - self.ref_lat) * meridian_radius(mid)
        x = self.ref_x + math.radians(lon - self.ref_lon) * parallel_radius(lat)
        return np.array([x, y, depth_km])

    def grid_to_geo(self, coords):
        if self.coord_sys == "spherical":
            lat, lon, depth = sph2geo_scalar(coords)
            return lat, lon, depth
        if _GEODESIC is not None:
            dx = coords[0] - self.ref_x
            dy = coords[1] - self.ref_y
            s_m = math.hypot(dx, dy) * 1000.0
            az = math.degrees(math.atan2(dx, dy))
            gd = _GEODESIC.Direct(self.ref_lat, self.ref_lon, az, s_m)
            return gd["lat2"], gd["lon2"], float(coords[2])
        # fallback inverse of the equirectangular mapping above
        lat = self.ref_lat + math.degrees(
            (coords[1] - self.ref_y) / meridian_radius(self.ref_lat))
        for _ in range(2):
            mid = 0.5 * (lat + self.ref_lat)
            lat = self.ref_lat + math.degrees(
                (coords[1] - self.ref_y) / meridian_radius(mid))
        lon = self.ref_lon + math.degrees(
            (coords[0] - self.ref_x) / parallel_radius(lat))
        return lat, lon, float(coords[2])

    def delta_to_grid(self, dlat_km, dlon_km, ddep_km):
        """Half-widths of the search box in grid units."""
        if self.coord_sys == "spherical":
            # (dr, dtheta, dphi) in km -> radians on the sphere
            r = pk_constants.EARTH_RADIUS
            return np.array([ddep_km, dlat_km / r, dlon_km / r])
        return np.array([dlon_km, dlat_km, ddep_km])

    def spatial_axes_are_xyz_enu(self):
        return self.coord_sys == "cartesian"


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
                               max_dist=None, nproc=None):
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

    IMPORTANT (segfault fix): the seiscomp Python bindings reference-count
    child objects (origin, arrivals, picks) through their parent
    EventParameters. If that parent is garbage-collected, the children
    become dangling pointers and the next attribute access (e.g.
    arrival.pickID()) segfaults. Earlier this function returned the
    origin/picks while dropping the parent, so we now (a) read into
    plain-Python structures here, while the parent is provably alive, and
    (b) still return the parent so the caller keeps it alive as a belt-
    and-braces measure.

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


def write_output(origin):
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
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="r+") as f:
        path = f.name
    ar = seiscomp.io.XMLArchive()
    ar.setFormattedOutput(True)
    if not ar.create(path):
        raise RuntimeError("Could not create output XML")
    ar.writeObject(origin)   # Origin directly, NOT wrapped in EventParameters
    ar.close()

    with open(path, "rb") as f:
        xml_bytes = f.read()

    # sanity checks before it ever reaches the C++ parent
    if not xml_bytes.lstrip().startswith(b"<?xml"):
        raise RuntimeError(
            "generated output does not begin with an XML declaration"
        )
    if b"<Origin" not in xml_bytes:
        raise RuntimeError("generated output contains no <Origin> element")

    _REAL_STDOUT.buffer.write(xml_bytes)
    _REAL_STDOUT.flush()

    try:
        os.unlink(path)
    except OSError:
        pass


# --------------------------------------------------------------------------
# main locator logic
# --------------------------------------------------------------------------
def main():
    # Capture the real stdout for the final XML, then redirect fd 1 to fd 2
    # so that anything pykonal / h5py / solvers / warnings might print goes
    # to stderr (visible in scolv logs) and can never corrupt the single
    # XML document the locext plugin reads from our stdout.
    global _REAL_STDOUT
    _REAL_STDOUT = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-dist", type=float, default=None)
    parser.add_argument("--fixed-depth", type=float, default=None)
    parser.add_argument("--ignore-initial-location", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        log(f"ignoring unknown args: {unknown}")

    with open(args.config) as f:
        cfg = json.load(f)

    transform = GeoTransform(cfg)
    method = cfg.get("method", "edt")
    alpha = float(cfg.get("alpha", 0.01))
    default_pick_error = float(cfg.get("default_pick_error", 0.1))
    delta_km = cfg.get("search_delta_km", [30.0, 30.0, 15.0])
    delta_t = float(cfg.get("search_delta_t", 10.0))
    nsamples = int(cfg.get("posterior_nsamples", 2048))
    # Fixed seed for posterior sampling so repeated relocations of the
    # same event give IDENTICAL uncertainties (the DE search is already
    # seeded separately). Set posterior_seed to null in the config for
    # independent draws each time (e.g. to gauge sampling variability).
    _pseed = cfg.get("posterior_seed", 4321)
    posterior_seed = None if _pseed is None else int(_pseed)

    stations = {}
    if cfg.get("stations_csv"):
        # CSV: NET,STA,lat,lon,elev_m — used for arrival distance/azimuth
        # and the --max-dist filter. Optional but recommended.
        with open(cfg["stations_csv"]) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                net, sta, lat, lon, elev = line.split(",")[:5]
                stations[(net, sta)] = (float(lat), float(lon), float(elev))

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
    for key, pid, t, terr, wt in pick_list:
        if key in arrivals:
            log(f"duplicate arrival for {key}; keeping first")
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
        pick_errors[key] = float(terr) if terr else default_pick_error
        pick_ids[key] = pid
        arrival_weights[key] = float(wt) if wt else 1.0

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
        _vax = 2 if transform.spatial_axes_are_xyz_enu() else 0
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
                max_dist=float(cfg.get("max_dist_km", None)),
                nproc=cfg.get("ensure_nproc"),
            )
        else:
            report = ensure_traveltimes(
                inv_path, requests, velocity_models,
                max_dist=float(cfg.get("max_dist_km", None)),
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
        locator.default_pick_error = default_pick_error
        # edt_reg is the one optional EDT knob (opt-in; 0 = pure EDT).
        if hasattr(locator, "edt_reg") and "edt_reg" in cfg:
            locator.edt_reg = float(cfg["edt_reg"])
        locator.add_arrivals(arrivals)
        locator.add_pick_errors(pick_errors)
        if grid_stations:
            locator.add_stations(grid_stations)

        _t_compute = _time.time()
        soln = locator.locate(initial, delta, alpha, method)
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
                _excluded.extend(_bad)
                locator.clear_arrivals()
                locator.add_arrivals(arrivals)
                locator.add_pick_errors(pick_errors)
                soln = locator.locate(initial, delta, alpha, method)
                residuals = locator.residuals(soln)
            if _excluded:
                log(f"rejected {len(_excluded)} outlier arrival(s) "
                    f"(cutoff {_cut:.2f} s): "
                    f"{', '.join('.'.join(k) for k in _excluded)}")

        quality = locator.quality(soln)
        posterior = None
        if method == "edt" and nsamples > 0:
            try:
                # Adaptive (contracting) posterior sampling.
                #
                # Sampling uniformly over the FULL search box is hopeless
                # when that box is large: the likelihood peak is a few km
                # wide, so almost no samples land in it and the weights
                # collapse onto a handful of points. Measured at a 300 km
                # half-width: effective sample size 2-4 out of 9000, with
                # sigma varying 6x between identical runs and drifting
                # smaller as the solution converged.
                #
                # So: sample, measure the scale, contract the proposal to a
                # few sigma around the solution, repeat. Each pass raises
                # the effective sample size by orders of magnitude and the
                # resulting covariance is stable and meaningful.
                # Contract GEOMETRICALLY rather than trusting sigma from
                # the first pass: when the effective sample size is ~2, the
                # covariance it returns is meaningless, so a sigma-driven
                # contraction would be built on noise. Shrinking by a fixed
                # factor each pass, floored at 4 sigma, converges reliably.
                _d = np.asarray(delta[:3], dtype=float).copy()
                for _pass in range(6):
                    posterior = locator.sample_posterior(
                        soln[:3], _d, nsamples=nsamples, nscatter=0,
                        seed=posterior_seed
                    )
                    _sig = np.sqrt(np.clip(
                        np.diag(posterior["covariance"]), 0, None))
                    _next = np.minimum(np.maximum(4.0 * _sig, 0.35 * _d), _d)
                    if np.all(_next > 0.95 * _d):
                        break            # converged: proposal already tight
                    _d = _next
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

        # 1-sigma uncertainties in km from the EDT posterior covariance,
        # computed here so they can be reported inline with the solution
        # and reused below when populating the origin. None if no posterior.
        lat_km = lon_km = dz = None
        if posterior is not None:
            _cov = posterior["covariance"]
            if transform.spatial_axes_are_xyz_enu():
                lon_km = math.sqrt(max(_cov[0, 0], 0.0))
                lat_km = math.sqrt(max(_cov[1, 1], 0.0))
                dz     = math.sqrt(max(_cov[2, 2], 0.0))
            else:
                _r0 = pk_constants.EARTH_RADIUS - dep
                _theta = math.radians(90.0 - lat)
                lat_km = math.sqrt(max(_cov[1, 1], 0.0)) * _r0
                lon_km = math.sqrt(max(_cov[2, 2], 0.0)) * _r0 * math.sin(_theta)
                dz     = math.sqrt(max(_cov[0, 0], 0.0))

    if lat_km is not None:
        _ess_txt = ""
        try:
            _ess_txt = f" ess={float(posterior['ess']):.0f}"
        except Exception:
            pass
        _unc = (f"lat {lat:.4f}\u00b0 \u00b1{lat_km:.2f} km, "
                f"lon {lon:.4f}\u00b0 \u00b1{lon_km:.2f} km, "
                f"depth {dep:.1f} \u00b1{dz:.2f} km{_ess_txt}")
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
        for key in sorted(residuals):
            _sta = stations.get(key[:2])
            if _sta is None:
                continue
            _ddeg, _ = gc_distance_azimuth(lat, lon, _sta[0], _sta[1])
            _d = _ddeg * 111.195          # degrees of arc -> km
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
    origin.setTime(seiscomp.datamodel.TimeQuantity(t0))
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
    # scolv reads Arrival.distance() unguarded (OriginLocatorView), so an
    # arrival added without it aborts the GUI with Core::ValueException.
    # Any station we cannot place geometrically is therefore dropped from
    # the output origin entirely, and announced below.
    used = 0
    dropped = []
    for key, pid in pick_ids.items():
        sta = stations.get(key[:2])
        if sta is None or not (np.isfinite(sta[0]) and np.isfinite(sta[1])):
            dropped.append(f"{key[0]}.{key[1]}.{key[2]}")
            continue

        arr = seiscomp.datamodel.Arrival()
        arr.setPickID(pid)
        arr.setPhase(seiscomp.datamodel.Phase(key[2]))
        dist, az = gc_distance_azimuth(lat, lon, sta[0], sta[1])
        arr.setDistance(dist)  # degrees of arc
        arr.setAzimuth(az)
        if key in residuals:
            arr.setTimeResidual(float(residuals[key]))
            arr.setWeight(arrival_weights.get(key, 1.0))
            arr.setTimeUsed(True)
            used += 1
        else:
            arr.setTimeResidual(0.0)
            arr.setWeight(0.0)
            arr.setTimeUsed(False)
        origin.add(arr)

    if dropped:
        log(f"dropped {len(dropped)} arrival(s) from the output origin — "
            f"no usable station coordinates: {', '.join(sorted(dropped))} "
            f"(add them to stations_csv)")

    # quality
    q = seiscomp.datamodel.OriginQuality()
    q.setUsedPhaseCount(used)
    q.setAssociatedPhaseCount(origin.arrivalCount())
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
        semi = posterior["ellipsoid"]["semi_axes"] * 1000.0  # km -> m
        axes = posterior["ellipsoid"]["axes"]

        ce = seiscomp.datamodel.ConfidenceEllipsoid()
        ce.setSemiMajorAxisLength(float(semi[0]))
        ce.setSemiIntermediateAxisLength(float(semi[1]))
        ce.setSemiMinorAxisLength(float(semi[2]))
        if transform.spatial_axes_are_xyz_enu():
            # axis vectors are (E, N, down-ish z): derive azimuth & plunge
            e, n, z = axes[0]
            ce.setMajorAxisAzimuth(math.degrees(math.atan2(e, n)) % 360.0)
            ce.setMajorAxisPlunge(math.degrees(math.atan2(z, math.hypot(e, n))))
            ce.setMajorAxisRotation(0.0)
        else:
            # spherical axes (r, theta, phi) ~ (up, south, east)
            r, th, ph = axes[0]
            ce.setMajorAxisAzimuth(math.degrees(math.atan2(ph, -th)) % 360.0)
            ce.setMajorAxisPlunge(math.degrees(math.atan2(-r, math.hypot(th, ph))))
            ce.setMajorAxisRotation(0.0)

        ou = seiscomp.datamodel.OriginUncertainty()
        ou.setConfidenceEllipsoid(ce)
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
        # Per-coordinate 1-sigma uncertainties in KM (SeisComP convention
        # for lat/lon/depth uncertainty is kilometers, not degrees). These
        # populate origin.latitude/longitude/depth .uncertainty(), which is
        # what scolv's location view displays. Axis mapping of the
        # posterior covariance:
        #   cartesian (E, N, depth): cov[0]=lon(E), cov[1]=lat(N), cov[2]=z
        #   spherical (r, theta, phi): cov[0]=depth(r),
        #       cov[1]=theta->lat, cov[2]=phi->lon (scaled by Earth radius,
        #       and by sin(theta) for the east-west metric)
        # lat_km / lon_km / dz were computed inline with the solution above
        # horizontalUncertainty (km): the larger horizontal 1-sigma
        ou.setHorizontalUncertainty(float(max(lat_km, lon_km)))
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

    write_output(origin)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
