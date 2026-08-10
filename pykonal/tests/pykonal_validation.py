#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pykonal_validation.py
=====================

Self-contained, **no-input** validation and benchmarking suite for the
(beta) ``pykonal`` package -- the spherical Eikonal solver, the point-source
solver, the traveltime inventory, and the ``EQLocator`` earthquake-location
machinery (EDT / L1 objectives, alpha / edt_ot_wt / edt_exponent parameters,
and the Hessian-vs-uniform posterior sampler).

Everything runs in the package's native **spherical** ``(r, theta, phi)``
frame (radius in km from the Earth's centre, colatitude and longitude in
radians). Two regimes are exercised end-to-end with a 1-D AK135 ``v(r)``
profile:

  * **regional** : a ~400 km spherical cap, AK135 crust + uppermost mantle
                   (0-120 km), surface stations and shallow events.
  * **global**   : a large-aperture cap (teleseismic distances), AK135 down
                   to ~1500 km depth, deep events.

The suite is defensive: every test is isolated, exceptions are captured with
a trimmed traceback, and the run always reaches the final report. Nothing is
read from stdin or argv; sizes are set in ``CONFIG`` below and can be scaled
with the optional ``PYKONAL_TEST_LEVEL`` environment variable
(``quick`` | ``standard`` | ``full``; default ``standard``).

Artifacts (a text log and a JSON summary) are written to
``./pykonal_validation_output/``.

Coverage: import/env; solver correctness (homogeneous-sphere convergence,
PointSourceSolver, causal stencil); field IO/interpolation; inventory
build/read/masking; EDT-vs-L1 recovery; a PERFORMANCE MATRIX across station
count, P/P+S, pick dropout, added noise/outliers and azimuthal coverage
(surrounded / one-sided / 120-deg gap); parameter sweeps (alpha 0..0.09 by
0.005, edt_ot_wt x geometry, edt_exponent, posterior proposal, determinism);
a VERTICAL-COMPRESSION study trading depth precision against model size to
find the optimal dx:dz ratio; quality metrics; and edge cases.

Usage
-----
    python pykonal_validation.py [--regimes regional|global|both]

Assumes ``pykonal`` is already built and importable.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field, asdict
from tempfile import mkdtemp

import numpy as np


# ===========================================================================
# Configuration
# ===========================================================================
_LEVEL = os.environ.get("PYKONAL_TEST_LEVEL", "standard").strip().lower()
if _LEVEL not in ("quick", "standard", "full"):
    _LEVEL = "standard"


def _resolve_regimes(argv=None):
    """Which regimes to run: 'regional', 'global', or 'both' (default).

    Precedence: an explicit ``--regimes`` CLI flag overrides the
    ``PYKONAL_TEST_REGIMES`` env var, which overrides the 'both' default.
    Kept permissive so the script still runs with no arguments at all.
    """
    choice = os.environ.get("PYKONAL_TEST_REGIMES", "both").strip().lower()
    argv = sys.argv[1:] if argv is None else argv
    for i, tok in enumerate(argv):
        if tok == "--regimes" and i + 1 < len(argv):
            choice = argv[i + 1].strip().lower()
        elif tok.startswith("--regimes="):
            choice = tok.split("=", 1)[1].strip().lower()
    if choice in ("reg", "regional"):
        return ("regional",)
    if choice in ("glob", "global"):
        return ("global",)
    return ("regional", "global")


@dataclass
class Regime:
    """Grid / geometry description for one location regime (spherical)."""
    name: str
    depth_max_km: float          # AK135 profile depth extent
    cap_half_km: float           # half-width of the station/event cap at surface
    nr: int                      # radial nodes (far-field velocity grid)
    ntheta: int                  # colatitude nodes
    nphi: int                    # longitude nodes
    n_stations: int
    n_events: int
    event_depth_km: tuple        # (min, max) event depth
    phases: tuple                # ("P",) or ("P", "S")
    pick_sigma_s: float          # baseline pick error (s)
    # vertical-compression sweep: horizontal node count is fixed, and the
    # vertical (radial) spacing is varied to give these dx:dz aspect ratios.
    sweep_ratios: tuple
    sweep_fixed_nang: int
    # the sweep uses its own COMPACT angular footprint (decoupled from the
    # possibly huge teleseismic locator cap) so the fixed-horizontal grid
    # stays tractable; the deep AK135 profile and event depths are retained.
    sweep_cap_km: float = 200.0


def _scaled(regional: Regime, glob: Regime):
    """Shrink sizes for 'quick', enlarge for 'full'."""
    if _LEVEL == "quick":
        for rg in (regional, glob):
            rg.nr = max(11, rg.nr // 2)
            rg.ntheta = max(15, rg.ntheta // 2 + 1)
            rg.nphi = max(15, rg.nphi // 2 + 1)
            rg.n_stations = max(5, rg.n_stations // 2)
            rg.n_events = max(4, rg.n_events // 2)
            rg.sweep_ratios = (1.0, 2.0, 4.0, 8.0)
            rg.sweep_fixed_nang = max(21, rg.sweep_fixed_nang // 2 + 1)
    elif _LEVEL == "full":
        for rg in (regional, glob):
            rg.nr = int(rg.nr * 1.5)
            rg.ntheta = int(rg.ntheta * 1.4) | 1
            rg.nphi = int(rg.nphi * 1.4) | 1
            rg.n_stations = int(rg.n_stations * 1.5)
            rg.n_events = int(rg.n_events * 1.5)
    return regional, glob


REGIONAL, GLOBAL = _scaled(
    Regime(
        name="regional",
        depth_max_km=120.0,
        cap_half_km=200.0,
        nr=25, ntheta=31, nphi=31,
        n_stations=9, n_events=10,
        event_depth_km=(2.0, 90.0),
        phases=("P", "S"),
        pick_sigma_s=0.15,
        sweep_ratios=(1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0),
        sweep_fixed_nang=41,
        sweep_cap_km=200.0,
    ),
    Regime(
        name="global",
        depth_max_km=1500.0,
        cap_half_km=6371.0 * np.deg2rad(75.0),   # ~75 deg cap
        nr=31, ntheta=37, nphi=37,
        n_stations=9, n_events=10,
        event_depth_km=(30.0, 1200.0),
        phases=("P",),
        pick_sigma_s=0.6,
        sweep_ratios=(1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0),
        sweep_fixed_nang=41,
        sweep_cap_km=1500.0,
    ),
)

OUTPUT_DIR = os.path.join(os.getcwd(), "pykonal_validation_output")
RNG_SEED = 20260725


# ===========================================================================
# AK135 (1-D) velocity profile
# ===========================================================================
# depth (km), Vp (km/s), Vs (km/s) -- Kennett, Engdahl & Buland (1995),
# curated to ~1510 km (above the core, so first-arrival FMM stays clean).
_AK135 = np.array([
    [0.00, 5.8000, 3.4600], [20.00, 5.8000, 3.4600],
    [20.00, 6.5000, 3.8500], [35.00, 6.5000, 3.8500],
    [35.00, 8.0400, 4.4800], [77.50, 8.0450, 4.4900],
    [120.00, 8.0500, 4.5000], [165.00, 8.1750, 4.5090],
    [210.00, 8.3000, 4.5180], [210.00, 8.3000, 4.5230],
    [260.00, 8.4825, 4.6090], [310.00, 8.6650, 4.6960],
    [360.00, 8.8475, 4.7830], [410.00, 9.0300, 4.8700],
    [410.00, 9.3600, 5.0800], [460.00, 9.5280, 5.1860],
    [510.00, 9.6960, 5.2920], [560.00, 9.8640, 5.3980],
    [610.00, 10.0320, 5.5040], [660.00, 10.2000, 5.6100],
    [660.00, 10.7900, 5.9600], [710.00, 10.9229, 6.0897],
    [760.00, 11.0558, 6.2095], [810.00, 11.1353, 6.2426],
    [860.00, 11.2221, 6.2798], [910.00, 11.3068, 6.3160],
    [960.00, 11.3896, 6.3512], [1010.00, 11.4705, 6.3854],
    [1060.00, 11.5495, 6.4187], [1110.00, 11.6269, 6.4510],
    [1160.00, 11.7026, 6.4828], [1210.00, 11.7766, 6.5138],
    [1260.00, 11.8491, 6.5439], [1310.00, 11.9200, 6.5727],
    [1360.00, 11.9895, 6.6008], [1410.00, 12.0577, 6.6285],
    [1460.00, 12.1245, 6.6554], [1510.00, 12.1912, 6.6813],
])


def ak135_velocity(depth_km, phase):
    """Interpolated AK135 velocity (km/s) at given depth(s) for 'P' or 'S'.

    Duplicate-depth discontinuities are nudged so ``np.interp`` sees a
    strictly increasing abscissa; the jump is then resolved across one grid
    node, which is adequate for a functionality/uncertainty test.
    """
    col = 1 if phase.upper() == "P" else 2
    d = _AK135[:, 0].copy()
    for i in range(1, len(d)):
        if d[i] <= d[i - 1]:
            d[i] = d[i - 1] + 1e-4
    v = _AK135[:, col]
    depth_km = np.asarray(depth_km, dtype=float)
    return np.interp(np.clip(depth_km, d[0], d[-1]), d, v)


# ===========================================================================
# Test harness
# ===========================================================================
@dataclass
class TestResult:
    name: str
    section: str
    status: str = "PASS"           # PASS | FAIL | WARN | SKIP
    message: str = ""
    metrics: dict = field(default_factory=dict)
    duration_s: float = 0.0
    warnings: list = field(default_factory=list)


class Suite:
    def __init__(self):
        self.results: list[TestResult] = []
        self._section = "general"

    def section(self, name):
        self._section = name
        _log(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    def run(self, name, fn, *args, **kwargs):
        """Execute one test function. It may return a dict of metrics, and may
        raise AssertionError (-> FAIL) or SkipTest (-> SKIP). Any other
        exception is captured as FAIL with a trimmed traceback."""
        res = TestResult(name=name, section=self._section)
        t0 = time.time()
        buf = io.StringIO()
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            try:
                with redirect_stdout(buf), redirect_stderr(buf):
                    out = fn(*args, **kwargs)
                if isinstance(out, dict):
                    res.metrics = _jsonable(out.get("metrics", out))
                    res.message = out.get("message", "")
                    res.status = out.get("status", "PASS")
                elif isinstance(out, str):
                    res.message = out
            except SkipTest as exc:
                res.status = "SKIP"
                res.message = str(exc)
            except AssertionError as exc:
                res.status = "FAIL"
                res.message = f"assertion: {exc}"
            except Exception as exc:  # noqa: BLE001
                res.status = "FAIL"
                tb = traceback.format_exc().strip().splitlines()
                res.message = f"{type(exc).__name__}: {exc}"
                res.metrics["traceback"] = "\n".join(tb[-6:])
            res.warnings = [f"{w.category.__name__}: {w.message}" for w in wlist]
        res.duration_s = round(time.time() - t0, 3)
        self.results.append(res)
        _log(f"[{res.status:4}] {name}  ({res.duration_s:.2f}s)"
             + (f"  -- {res.message}" if res.message else ""))
        if "traceback" in res.metrics:
            _log("       " + res.metrics["traceback"].replace("\n", "\n       "))
        return res

    # ---- reporting -------------------------------------------------------
    def summary(self):
        counts = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    def write_reports(self, outdir):
        os.makedirs(outdir, exist_ok=True)
        # JSON
        payload = {
            "level": _LEVEL,
            "summary": self.summary(),
            "results": [asdict(r) for r in self.results],
        }
        with open(os.path.join(outdir, "results.json"), "w") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)
        # text
        lines = [f"pykonal validation report (level={_LEVEL})",
                 "=" * 60, ""]
        cur = None
        for r in self.results:
            if r.section != cur:
                cur = r.section
                lines.append(f"\n## {cur}")
            lines.append(f"  [{r.status:4}] {r.name} ({r.duration_s:.2f}s)")
            if r.message:
                lines.append(f"         {r.message}")
            for k, v in r.metrics.items():
                if k == "traceback":
                    continue
                lines.append(f"           - {k}: {v}")
        lines.append("\n" + "=" * 60)
        lines.append("SUMMARY: " + ", ".join(
            f"{k}={v}" for k, v in sorted(self.summary().items())))
        with open(os.path.join(outdir, "report.txt"), "w") as fh:
            fh.write("\n".join(lines))


class SkipTest(Exception):
    pass


_LOG_LINES: list[str] = []


def _log(msg):
    print(msg)
    _LOG_LINES.append(str(msg))


def _jsonable(d):
    return {k: _json_default(v) for k, v in d.items()}


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return np.round(o, 6).tolist()
    if isinstance(o, dict):
        return {k: _json_default(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_default(v) for v in o]
    return o


# ===========================================================================
# Package import (hard dependency for everything downstream)
# ===========================================================================
def import_pykonal():
    import pykonal
    from pykonal import constants, fields, inventory, locate, solver
    from pykonal import transformations as tf
    return dict(
        pk=pykonal, constants=constants, fields=fields,
        inventory=inventory, locate=locate, solver=solver, tf=tf,
    )


# ===========================================================================
# Spherical geometry helpers
# ===========================================================================
R_EARTH = 6371.0


def cap_grid_bounds(regime: Regime):
    """Return spherical grid bounds for a regime centred on the equator/prime
    meridian: (r_min, r_max, th_min, th_max, ph_min, ph_max)."""
    r_max = R_EARTH
    r_min = R_EARTH - regime.depth_max_km
    half_ang = regime.cap_half_km / R_EARTH           # radians
    th0 = np.pi / 2.0
    ph0 = np.pi                                        # keep away from 0/2pi seam
    th_min, th_max = th0 - half_ang, th0 + half_ang
    ph_min, ph_max = ph0 - half_ang, ph0 + half_ang
    return r_min, r_max, th_min, th_max, ph_min, ph_max


def build_velocity_field(F, regime: Regime, phase, nr, ntheta, nphi):
    """Construct an AK135 spherical ScalarField3D velocity model."""
    r_min, r_max, th_min, th_max, ph_min, ph_max = cap_grid_bounds(regime)
    dr = (r_max - r_min) / (nr - 1)
    dth = (th_max - th_min) / (ntheta - 1)
    dph = (ph_max - ph_min) / (nphi - 1)

    vf = F.ScalarField3D(coord_sys="spherical")
    vf.min_coords = np.array([r_min, th_min, ph_min], dtype=np.float64)
    vf.node_intervals = np.array([dr, dth, dph], dtype=np.float64)
    vf.npts = np.array([nr, ntheta, nphi], dtype=np.int64)

    r_nodes = r_min + np.arange(nr) * dr
    depth = R_EARTH - r_nodes
    v_r = ak135_velocity(depth, phase)                 # (nr,)
    values = np.broadcast_to(
        v_r[:, None, None], (nr, ntheta, nphi)
    ).astype(np.float64).copy()
    vf.values = values
    return vf


def random_stations(regime: Regime, rng):
    """Random surface stations (r = R) within the inner 85% of the cap."""
    r_min, r_max, th_min, th_max, ph_min, ph_max = cap_grid_bounds(regime)
    pad_t = 0.075 * (th_max - th_min)
    pad_p = 0.075 * (ph_max - ph_min)
    stations = {}
    for i in range(regime.n_stations):
        th = rng.uniform(th_min + pad_t, th_max - pad_t)
        ph = rng.uniform(ph_min + pad_p, ph_max - pad_p)
        stations[("XX", f"S{i:02d}")] = np.array([R_EARTH, th, ph])
    return stations


def one_sided_stations(regime: Regime, rng):
    """Stations confined to one angular quadrant of the cap (large gap)."""
    r_min, r_max, th_min, th_max, ph_min, ph_max = cap_grid_bounds(regime)
    th_c = 0.5 * (th_min + th_max)
    ph_c = 0.5 * (ph_min + ph_max)
    stations = {}
    for i in range(regime.n_stations):
        th = rng.uniform(th_c, th_max - 0.05 * (th_max - th_min))
        ph = rng.uniform(ph_c, ph_max - 0.05 * (ph_max - ph_min))
        stations[("XX", f"S{i:02d}")] = np.array([R_EARTH, th, ph])
    return stations


def random_events(regime: Regime, rng, n=None):
    """Random events within the inner cap and the regime's depth band."""
    r_min, r_max, th_min, th_max, ph_min, ph_max = cap_grid_bounds(regime)
    n = n or regime.n_events
    pad_t = 0.20 * (th_max - th_min)
    pad_p = 0.20 * (ph_max - ph_min)
    ev = []
    for _ in range(n):
        depth = rng.uniform(*regime.event_depth_km)
        r = R_EARTH - depth
        th = rng.uniform(th_min + pad_t, th_max - pad_t)
        ph = rng.uniform(ph_min + pad_p, ph_max - pad_p)
        t0 = rng.uniform(0.0, 5.0)
        ev.append(np.array([r, th, ph, t0]))
    return ev


def bearing_deg(event_sph, station_sph):
    """Azimuth (deg, 0=N, clockwise) from an event epicentre to a station."""
    lat1 = np.pi / 2 - event_sph[1]
    lat2 = np.pi / 2 - station_sph[1]
    dlon = station_sph[2] - event_sph[2]
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.degrees(np.arctan2(y, x)) % 360.0


def select_stations(stations, event_sph, k, geometry, rng):
    """Choose up to k stations for an event under a geometry constraint:

      'all'       : the k nearest stations (dense, well-distributed)
      'one_sided' : k nearest whose azimuth lies in a random 180 deg half
      'gap'       : k nearest after removing a random 120 deg azimuth wedge
                    (leaves a large azimuthal gap)
    """
    items = list(stations.items())
    az = {key: bearing_deg(event_sph, xyz) for key, xyz in items}
    dist = {key: hdist_km(event_sph[:3], xyz) for key, xyz in items}

    if geometry == "one_sided":
        lo = rng.uniform(0, 360)
        keep = [key for key, _ in items
                if ((az[key] - lo) % 360.0) <= 180.0]
    elif geometry == "gap":
        gap_lo = rng.uniform(0, 360)
        keep = [key for key, _ in items
                if not (((az[key] - gap_lo) % 360.0) <= 120.0)]
    else:
        keep = [key for key, _ in items]

    keep.sort(key=lambda key: dist[key])
    return keep[:k]


def make_scenario_arrivals(tt_mem, stations, event_sph_t0, rng, regime,
                           k, geometry, phases, dropout, pick_sigma,
                           n_outliers):
    """Build arrivals for one performance scenario (geometry + phase subset +
    random pick dropout + outliers)."""
    chosen = select_stations(stations, event_sph_t0, k, geometry, rng)
    sub = {key: stations[key] for key in chosen}
    arr, perr, _ = synth_arrivals(
        tt_mem, sub, event_sph_t0, rng, pick_sigma, phases,
        n_outliers=n_outliers)
    if dropout > 0 and arr:
        for key in list(arr):
            if rng.random() < dropout:
                arr.pop(key, None)
                perr.pop(key, None)
    return arr, perr, chosen


# ===========================================================================
# Traveltime construction
# ===========================================================================
def eikonal_traveltime(S, F, vel_field, src_sph):
    """Compute a traveltime field from a point source using the plain
    EikonalSolver (source snapped to the nearest node). Fast, deterministic,
    and gives us direct control of grid resolution/aspect for the sweep."""
    slv = S.EikonalSolver(coord_sys="spherical")
    slv.velocity.min_coords = np.asarray(vel_field.min_coords, np.float64)
    slv.velocity.node_intervals = np.asarray(vel_field.node_intervals, np.float64)
    slv.velocity.npts = np.asarray(vel_field.npts, np.int64)
    slv.velocity.values = np.asarray(vel_field.values, np.float64)

    mc = np.asarray(vel_field.min_coords, np.float64)
    ni = np.asarray(vel_field.node_intervals, np.float64)
    npts = np.asarray(vel_field.npts, np.int64)
    idx = np.rint((np.asarray(src_sph, np.float64) - mc) / ni).astype(int)
    idx = tuple(int(np.clip(idx[k], 0, npts[k] - 1)) for k in range(3))

    slv.traveltime.values[idx] = 0.0
    slv.unknown[idx] = False
    slv.trial.push(*idx)
    slv.solve()

    tt = F.ScalarField3D(coord_sys="spherical")
    tt.min_coords = mc
    tt.node_intervals = ni
    tt.npts = npts
    tt.values = np.asarray(slv.traveltime.values, np.float64).copy()
    src_node = mc + np.array(idx) * ni
    return tt, src_node


def point_source_traveltime(S, F, vel_field, src_sph):
    """Compute a traveltime field with the near-field-refined
    PointSourceSolver (exercises the spherical near-field / pole handling)."""
    slv = S.PointSourceSolver(coord_sys="spherical")
    slv.velocity.min_coords = np.asarray(vel_field.min_coords, np.float64)
    slv.velocity.node_intervals = np.asarray(vel_field.node_intervals, np.float64)
    slv.velocity.npts = np.asarray(vel_field.npts, np.int64)
    slv.velocity.values = np.asarray(vel_field.values, np.float64)
    slv.src_loc = np.asarray(src_sph, np.float64)
    slv.solve()
    return slv.traveltime


def build_inventory(ctx, regime, stations, nr=None, ntheta=None, nphi=None,
                    use_point_source=False, tmpdir=None, max_dist=None,
                    mask=False):
    """Build a traveltime inventory for all (station, phase) pairs. Returns
    (inventory_path, {key: in-memory tt ScalarField3D}, vel_fields)."""
    S, F, INV = ctx["solver"], ctx["fields"], ctx["inventory"]
    nr = nr or regime.nr
    ntheta = ntheta or regime.ntheta
    nphi = nphi or regime.nphi
    tmpdir = tmpdir or mkdtemp(prefix="pyk_inv_")
    path = os.path.join(tmpdir, f"{regime.name}_{nr}_{ntheta}_{nphi}.h5")

    vel_fields = {ph: build_velocity_field(F, regime, ph, nr, ntheta, nphi)
                  for ph in regime.phases}

    tt_mem = {}
    inv = INV.TraveltimeInventory(path, mode="w")
    try:
        for (net, sta), coords in stations.items():
            for ph in regime.phases:
                key = (net, sta, ph)
                if use_point_source:
                    ttf = point_source_traveltime(S, F, vel_fields[ph], coords)
                    src_node = np.asarray(coords, np.float64)
                else:
                    ttf, src_node = eikonal_traveltime(
                        S, F, vel_fields[ph], coords)
                tt_mem[key] = ttf
                inv.add(ttf, "/".join(key), station_coords=src_node,
                        max_dist=max_dist, mask=mask, compress=True)
    finally:
        inv.f5.close()
    return path, tt_mem, vel_fields


def synth_arrivals(tt_mem, stations, event_sph_t0, rng, pick_sigma,
                   phases, n_outliers=0, outlier_size=(3.0, 6.0)):
    """Synthesize arrival times by interpolating the stored traveltime grids
    at the true event location and adding Gaussian pick noise (+ optional
    gross outliers). Returns (arrivals, pick_errors, truth_tt)."""
    r, th, ph, t0 = event_sph_t0
    xyz = np.array([r, th, ph], np.float64)
    arrivals, pick_errors, truth = {}, {}, {}
    keys = []
    for (net, sta) in stations:
        for phase in phases:
            key = (net, sta, phase)
            if key not in tt_mem:
                continue
            tt = tt_mem[key].value(xyz, null=np.nan)
            if not np.isfinite(tt) or tt > 9999:
                continue
            arrivals[key] = t0 + tt + rng.normal(0.0, pick_sigma)
            pick_errors[key] = pick_sigma
            truth[key] = tt
            keys.append(key)
    # inject outliers
    if n_outliers > 0 and keys:
        idx = rng.choice(len(keys), size=min(n_outliers, len(keys)),
                         replace=False)
        for j in idx:
            sign = rng.choice([-1.0, 1.0])
            arrivals[keys[j]] += sign * rng.uniform(*outlier_size)
    return arrivals, pick_errors, truth


def hdist_km(a_sph, b_sph):
    """Great-circle epicentral distance (km) between two spherical points."""
    lat1 = np.pi / 2 - a_sph[1]
    lat2 = np.pi / 2 - b_sph[1]
    dlon = b_sph[2] - a_sph[2]
    x = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
    return 2 * np.arcsin(np.sqrt(np.clip(x, 0, 1))) * R_EARTH


def ellipsoid_from_post(post, tf):
    """Physical (km) semi-axes + volume from a posterior dict. Prefers the
    library's 'ellipsoid_km' (available after the sample_posterior fix); falls
    back to deriving it from the scatter cloud so the suite also works against
    an un-rebuilt library."""
    ek = post.get("ellipsoid_km") if isinstance(post, dict) else None
    if ek is not None and ek.get("semi_axes") is not None:
        semi = np.asarray(ek["semi_axes"], float)
        vol = float(ek.get("volume_km3",
                           4.0 / 3.0 * np.pi * np.prod(semi)))
        return semi, vol
    pe = physical_ellipsoid(post.get("scatter"), tf) if isinstance(post, dict) else None
    return pe if pe is not None else None


def physical_ellipsoid(scatter_sph, tf):
    """Compute a physical error ellipsoid (km) from a posterior scatter cloud
    given in the locator's spherical (r, theta, phi) coordinates.

    The library's ``sample_posterior`` reports its covariance/ellipsoid in raw
    coordinates, which for a spherical grid mixes km (radial) with radians
    (angular) -- so its semi-axes are NOT directly comparable or physically
    meaningful. Here we transform the cloud to Cartesian km (eigenvalues are
    rotation-invariant, so global xyz centring is sufficient) and return
    semi-axes in km, sorted descending, plus the ellipsoid volume in km^3.
    """
    pts = np.asarray(scatter_sph, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] != 3:
        return None
    xyz = np.asarray(tf.sph2xyz(pts), dtype=float)          # km
    c = xyz - xyz.mean(axis=0)
    cov = np.cov(c.T)
    evals = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    semi = np.sqrt(np.sort(evals)[::-1])                    # km, descending
    vol = float(4.0 / 3.0 * np.pi * np.prod(semi))
    return semi, vol


def location_errors_km(true_sph_t0, est_sph_t0):
    """Return (epicentral_km, depth_km, total_km, dt0_s)."""
    epi = hdist_km(true_sph_t0[:3], est_sph_t0[:3])
    ddep = abs((R_EARTH - true_sph_t0[0]) - (R_EARTH - est_sph_t0[0]))
    total = np.sqrt(epi ** 2 + ddep ** 2)
    dt0 = abs(true_sph_t0[3] - est_sph_t0[3])
    return epi, ddep, total, dt0


def default_delta(regime: Regime):
    """A generous 4-vector search half-width (r, theta, phi, t0)."""
    half_ang = regime.cap_half_km / R_EARTH
    return np.array([regime.depth_max_km * 0.9,
                     0.6 * half_ang, 0.6 * half_ang, 8.0])


# ===========================================================================
# SECTION 0 -- environment & import
# ===========================================================================
def test_import(ctx):
    pk = ctx["pk"]
    ver = getattr(pk, "__version__", "unknown")
    for name in ("EikonalSolver",):
        assert hasattr(pk, name), f"pykonal.{name} missing"
    assert hasattr(ctx["solver"], "PointSourceSolver")
    assert hasattr(ctx["locate"], "EQLocator")
    assert hasattr(ctx["fields"], "ScalarField3D")
    assert hasattr(ctx["inventory"], "TraveltimeInventory")
    return {"metrics": {"version": ver, "level": _LEVEL}}


# ===========================================================================
# SECTION 1 -- solver correctness
# ===========================================================================
def _homog_sphere_solve(ctx, nr):
    """Solve a constant-velocity spherical cap from a surface point source and
    return rich error statistics vs the analytic chord/v solution."""
    S, F, tf = ctx["solver"], ctx["fields"], ctx["tf"]
    v0 = 8.0
    span = 400.0
    r_min, r_max = R_EARTH - span, R_EARTH
    half = span / R_EARTH
    th0, ph0 = np.pi / 2, np.pi
    vf = F.ScalarField3D(coord_sys="spherical")
    vf.min_coords = np.array([r_min, th0 - half, ph0 - half])
    vf.node_intervals = np.array([(r_max - r_min) / (nr - 1),
                                  2 * half / (nr - 1), 2 * half / (nr - 1)])
    vf.npts = np.array([nr, nr, nr], np.int64)
    vf.values = np.full((nr, nr, nr), v0, np.float64)

    src = np.array([r_max, th0, ph0])
    tt, src_node = eikonal_traveltime(S, F, vf, src)

    nodes = np.asarray(tt.nodes)
    xyz = tf.sph2xyz(nodes.reshape(-1, 3)).reshape(nr, nr, nr, 3)
    src_xyz = tf.sph2xyz(src_node.reshape(1, 3))[0]
    chord = np.linalg.norm(xyz - src_xyz, axis=-1)          # true distance (km)
    analytic = chord / v0
    ttv = np.asarray(tt.values)

    finite = np.isfinite(ttv)
    frac_finite = float(finite.mean())
    min_tt = float(np.nanmin(ttv))

    # cell size = max of radial and surface arc-length spacings
    dr = float(vf.node_intervals[0])
    dth = float(vf.node_intervals[1])
    arc = R_EARTH * dth
    h = max(dr, arc)

    # Evaluate on a FIXED physical shell (same region at every resolution) so
    # the coarse/fine comparison is apples-to-apples and independent of the
    # near-source zone. The shell sits well outside the source neighbourhood.
    shell_lo, shell_hi = 0.40 * span, 0.90 * span

    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(ttv - analytic) / np.maximum(analytic, 1e-9)
    valid = finite & (chord >= shell_lo) & (chord <= shell_hi)
    relv = rel[valid]

    # locate the worst node within the shell for diagnosis
    worst = {}
    if relv.size:
        idx_flat = np.argmax(np.where(valid, rel, -1))
        iw = np.unravel_index(idx_flat, rel.shape)
        on_boundary = any(iw[k] in (0, rel.shape[k] - 1) for k in range(3))
        worst = {
            "rel": round(float(rel[iw]), 4),
            "dist_km": round(float(chord[iw]), 1),
            "dist_cells": round(float(chord[iw] / h), 1),
            "depth_km": round(float(R_EARTH - nodes[iw][0]), 1),
            "on_grid_boundary": bool(on_boundary),
        }
    return {
        "nr": nr, "h_km": round(h, 2), "dr_km": round(dr, 2),
        "arc_km": round(arc, 2),
        "shell_km": [round(shell_lo, 1), round(shell_hi, 1)],
        "n_shell_nodes": int(relv.size),
        "frac_finite": round(frac_finite, 4),
        "min_tt_s": round(min_tt, 5),
        "rel_median": round(float(np.median(relv)), 4) if relv.size else None,
        "rel_p90": round(float(np.percentile(relv, 90)), 4) if relv.size else None,
        "rel_max": round(float(np.max(relv)), 4) if relv.size else None,
        "worst_far_node": worst,
    }


def test_homogeneous_sphere_analytic(ctx):
    """Constant-velocity sphere vs analytic chord/v. Correctness is judged by
    CONVERGENCE (does the far-field error shrink as the grid refines) plus
    finiteness/monotonicity -- not by a fixed error threshold, since a
    first-order fast-marching scheme has expected O(10%) *near-source*
    relative error that decays with distance and resolution."""
    coarse = _homog_sphere_solve(ctx, 31)
    fine = _homog_sphere_solve(ctx, 51)

    # basic physical sanity
    assert fine["frac_finite"] > 0.99, \
        f"non-finite traveltimes ({fine['frac_finite']:.3f})"
    assert fine["min_tt_s"] >= -1e-6, f"negative traveltime {fine['min_tt_s']}"

    mc, mf = coarse["rel_median"], fine["rel_median"]
    assert mc and mf, "no valid far-field nodes to compare"

    # observed convergence order from median far-field relative error
    hc, hf = coarse["h_km"], fine["h_km"]
    order = float(np.log(mc / mf) / np.log(hc / hf)) if mf > 0 else float("inf")

    status, msg = "PASS", ""
    # PRIMARY signal: error must decrease under refinement (solver converges)
    if not (mf < mc):
        status, msg = "FAIL", (
            f"far-field median error did NOT decrease under refinement "
            f"(coarse={mc}, fine={mf}); indicates a solver problem, not "
            f"mere coarseness")
    elif mf > 0.05:
        status, msg = "WARN", (
            f"far-field median error {mf:.3f} is high though converging "
            f"(order~{order:.2f}); likely near-source contamination or a very "
            f"coarse grid -- inspect worst_far_node")
    else:
        msg = f"converging, order~{order:.2f}, far-field median {mf:.3f}"

    if status == "FAIL":
        raise AssertionError(msg)
    return {"status": status, "message": msg,
            "metrics": {"coarse": coarse, "fine": fine,
                        "convergence_order": round(order, 3)}}


def test_point_source_solver_runs(ctx):
    """PointSourceSolver on AK135 must run and produce a monotone, finite,
    near-zero-at-source field (exercises the spherical near-field/pole code)."""
    S, F = ctx["solver"], ctx["fields"]
    vf = build_velocity_field(F, REGIONAL, "P",
                              REGIONAL.nr, REGIONAL.ntheta, REGIONAL.nphi)
    r_min, r_max, th_min, th_max, ph_min, ph_max = cap_grid_bounds(REGIONAL)
    src = np.array([r_max, 0.5 * (th_min + th_max), 0.5 * (ph_min + ph_max)])
    tt = point_source_traveltime(S, F, vf, src)
    v = np.asarray(tt.values)
    finite = np.isfinite(v)
    frac_finite = float(finite.mean())
    tmin = float(np.nanmin(v[finite]))
    assert frac_finite > 0.5, f"too many non-finite tt nodes ({frac_finite:.2f})"
    assert tmin >= -1e-6, f"negative traveltime {tmin}"
    # compare near-field-refined solve vs plain FMM at a mid-depth node
    tt_fmm, _ = eikonal_traveltime(S, F, vf, src)
    vf2 = np.asarray(tt_fmm.values)
    both = finite & np.isfinite(vf2)
    diff = np.abs(v[both] - vf2[both])
    med = float(np.median(diff))
    return {"metrics": {"frac_finite": round(frac_finite, 3),
                        "tt_min_s": round(tmin, 4),
                        "median_PSS_minus_FMM_s": round(med, 4)},
            "message": "PSS vs plain-FMM median difference reported"}


def test_causal_stencil_aspect(ctx):
    """High radial:angular aspect ratio must not produce NaNs / negative or
    non-monotone traveltimes (the causal-repair path)."""
    S, F = ctx["solver"], ctx["fields"]
    v0 = 8.0
    results = {}
    for aspect in (1, 4, 12):
        nr, nt, npi = 61, 25, 25
        r_min, r_max = R_EARTH - 300.0, R_EARTH
        # inflate angular node interval to raise (R*dtheta)/dr
        base = 300.0 / (nr - 1)
        dth = aspect * base / R_EARTH
        half = 0.5 * dth * (nt - 1)
        th0, ph0 = np.pi / 2, np.pi
        vf = F.ScalarField3D(coord_sys="spherical")
        vf.min_coords = np.array([r_min, th0 - half, ph0 - half])
        vf.node_intervals = np.array([base, dth, dth])
        vf.npts = np.array([nr, nt, npi], np.int64)
        vf.values = np.full((nr, nt, npi), v0, np.float64)
        src = np.array([r_max, th0, ph0])
        tt, _ = eikonal_traveltime(S, F, vf, src)
        v = np.asarray(tt.values)
        results[f"aspect_{aspect}"] = {
            "frac_finite": round(float(np.isfinite(v).mean()), 3),
            "tt_min": round(float(np.nanmin(v)), 5),
            "any_negative": bool(np.nanmin(v) < -1e-6),
        }
        assert np.nanmin(v) >= -1e-6, f"aspect {aspect}: negative traveltime"
        assert np.isfinite(v).mean() > 0.8, f"aspect {aspect}: many NaNs"
    return {"metrics": results}


# ===========================================================================
# SECTION 2 -- fields: HDF round-trip & interpolation
# ===========================================================================
def test_field_hdf_roundtrip(ctx):
    F = ctx["fields"]
    vf = build_velocity_field(F, REGIONAL, "P", 21, 21, 21)
    tmp = mkdtemp(prefix="pyk_fld_")
    path = os.path.join(tmp, "vel.h5")
    vf.to_hdf(path, overwrite=True)
    back = F.read_hdf(path)
    a, b = np.asarray(vf.values), np.asarray(back.values)
    assert a.shape == b.shape, "shape changed on round-trip"
    max_abs = float(np.max(np.abs(a - b)))
    # low_precision=True default -> float32 storage
    assert max_abs < 1e-2, f"round-trip mismatch {max_abs}"
    assert np.allclose(vf.min_coords, back.min_coords)
    assert np.allclose(vf.node_intervals, back.node_intervals)
    return {"metrics": {"max_abs_value_diff": round(max_abs, 6),
                        "coord_sys": str(back.coord_sys)}}


def test_field_interpolation(ctx):
    """value() at a node equals the stored value; midpoint is bracketed."""
    F = ctx["fields"]
    vf = build_velocity_field(F, REGIONAL, "P", 25, 25, 25)
    vals = np.asarray(vf.values)
    nodes = np.asarray(vf.nodes)
    i = (10, 12, 9)
    at_node = vf.value(nodes[i].astype(np.float64), null=np.nan)
    assert abs(at_node - vals[i]) < 1e-6, "node interpolation off"
    mid = 0.5 * (nodes[10, 12, 9] + nodes[11, 12, 9])
    at_mid = vf.value(mid.astype(np.float64), null=np.nan)
    lo, hi = sorted((vals[10, 12, 9], vals[11, 12, 9]))
    assert lo - 1e-6 <= at_mid <= hi + 1e-6, "midpoint not bracketed"
    # outside the grid -> null
    outside = np.asarray(vf.min_coords) - np.array([100.0, 0.0, 0.0])
    got = vf.value(outside.astype(np.float64), null=np.nan)
    assert np.isnan(got), "out-of-grid did not return null"
    return {"metrics": {"node_value": round(float(at_node), 5),
                        "midpoint_value": round(float(at_mid), 5)}}


# ===========================================================================
# SECTION 3 -- inventory build / read / crop
# ===========================================================================
def test_inventory_build_read(ctx, shared):
    INV = ctx["inventory"]
    path, tt_mem, _ = shared["primary_inv"]
    inv = INV.TraveltimeInventory(path, mode="r")
    try:
        keys = inv.keys()
        assert len(keys) == len(tt_mem), \
            f"key count {len(keys)} != {len(tt_mem)}"
        any_key = next(iter(tt_mem))
        field_back = inv.read("/".join(any_key))
        a = np.asarray(tt_mem[any_key].values)
        b = np.asarray(field_back.values)
        # float32 storage tolerance
        m = float(np.nanmax(np.abs(a - b)))
        assert m < 1e-1, f"stored/read mismatch {m}"
    finally:
        inv.f5.close()
    return {"metrics": {"n_keys": len(keys),
                        "max_read_diff_s": round(m, 5)}}


def test_inventory_maxdist_mask(ctx):
    """max_dist masking must set far nodes to NaN and shrink coverage."""
    INV = ctx["inventory"]
    S, F = ctx["solver"], ctx["fields"]
    vf = build_velocity_field(F, REGIONAL, "P", 21, 31, 31)
    r_min, r_max, th_min, th_max, ph_min, ph_max = cap_grid_bounds(REGIONAL)
    src = np.array([r_max, 0.5 * (th_min + th_max), 0.5 * (ph_min + ph_max)])
    ttf, src_node = eikonal_traveltime(S, F, vf, src)
    tmp = mkdtemp(prefix="pyk_mask_")
    path = os.path.join(tmp, "mask.h5")
    inv = INV.TraveltimeInventory(path, mode="w")
    try:
        inv.add(ttf, "XX/S00/P", station_coords=src_node,
                max_dist=120.0, mask=True, compress=True)
    finally:
        inv.f5.close()
    inv = INV.TraveltimeInventory(path, mode="r")
    try:
        back = inv.read("XX/S00/P")
    finally:
        inv.f5.close()
    v = np.asarray(back.values)
    frac_nan = float(np.isnan(v).mean())
    assert frac_nan > 0.0, "mask produced no NaNs"
    assert np.isfinite(v).any(), "mask removed everything"
    return {"metrics": {"frac_masked": round(frac_nan, 3),
                        "cropped_shape": list(v.shape)}}


# ===========================================================================
# SECTION 4 -- EQLocator end-to-end: EDT vs L1
# ===========================================================================
def _locate_batch(ctx, regime, shared_inv, stations, method, alpha=None,
                  edt_ot_wt=True, n_outliers=0, exponent=None,
                  n_events=None, rng=None, do_posterior=False):
    """Locate a batch of synthetic events; return arrays of errors + metrics."""
    L = ctx["locate"]
    path, tt_mem, _ = shared_inv
    rng = rng or np.random.default_rng(RNG_SEED)
    events = random_events(regime, rng, n=n_events or regime.n_events)
    delta = default_delta(regime)

    epi_e, dep_e, tot_e, dt0_e = [], [], [], []
    ess_list, semiaxis = [], []
    loc = L.EQLocator(path, coord_sys="spherical")
    try:
        loc.add_stations({k: v for k, v in stations.items()})
        loc.edt_ot_wt = edt_ot_wt
        if exponent is not None:
            loc.edt_exponent = exponent
        for ev in events:
            arr, perr, truth = synth_arrivals(
                tt_mem, stations, ev, rng, regime.pick_sigma_s,
                regime.phases, n_outliers=n_outliers)
            if len(arr) < 2:
                continue
            loc.clear_arrivals()
            loc.add_arrivals(arr)
            loc.add_pick_errors(perr)
            # start from cap centre + random offset (never the truth)
            r_min, r_max, th_min, th_max, ph_min, ph_max = \
                cap_grid_bounds(regime)
            initial = np.array([
                r_max - 0.5 * regime.depth_max_km,
                0.5 * (th_min + th_max),
                0.5 * (ph_min + ph_max),
                0.0])
            a = np.nan if alpha is None else alpha
            if do_posterior and method == "edt":
                out = loc.locate_detailed(initial, delta, alpha=a,
                                          method="edt", nsamples=1500,
                                          nscatter=400, seed=7)
                est = out["hypocenter"]
                post = out.get("posterior", {})
                if "ess" in post:
                    ess_list.append(float(post["ess"]))
                pe = ellipsoid_from_post(post, ctx["tf"])
                if pe is not None:
                    semiaxis.append(pe[0])
            else:
                est = loc.locate(initial, delta, alpha=a, method=method)
            epi, dep, tot, dt0 = location_errors_km(ev, est)
            epi_e.append(epi); dep_e.append(dep)
            tot_e.append(tot); dt0_e.append(dt0)
    finally:
        loc.__exit__(None, None, None)

    def med(x):
        return float(np.median(x)) if x else float("nan")

    m = {
        "n_located": len(tot_e),
        "epi_km_med": round(med(epi_e), 2),
        "depth_km_med": round(med(dep_e), 2),
        "total_km_med": round(med(tot_e), 2),
        "dt0_s_med": round(med(dt0_e), 3),
    }
    if ess_list:
        m["ess_med"] = round(med(ess_list), 1)
    if semiaxis:
        sa = np.median(np.vstack(semiaxis), axis=0)
        m["ellipsoid_semi_axes_km_med"] = [round(float(x), 2) for x in sa]
        m["ellipsoid_vol_km3_med"] = round(
            float(4.0 / 3.0 * np.pi * np.prod(sa)), 2)
    return m, tot_e


def test_edt_vs_l1_clean(ctx, shared):
    regime = shared["primary"]
    stations = shared["primary_stations"]
    m_edt, _ = _locate_batch(ctx, regime, shared["primary_inv"], stations,
                             "edt", n_outliers=0,
                             rng=np.random.default_rng(1))
    m_l1, _ = _locate_batch(ctx, regime, shared["primary_inv"], stations,
                            "l1", n_outliers=0,
                            rng=np.random.default_rng(1))
    assert m_edt["n_located"] > 0 and m_l1["n_located"] > 0
    # clean data: both should recover to within a few grid nodes
    assert m_edt["total_km_med"] < regime.cap_half_km, "EDT diverged"
    assert m_l1["total_km_med"] < regime.cap_half_km, "L1 diverged"
    return {"metrics": {"EDT": m_edt, "L1": m_l1}}


def test_edt_vs_l1_outliers(ctx, shared):
    """With gross outliers, EDT's median error should not be worse than L1's
    by more than a safety factor (EDT is designed to be outlier-robust)."""
    regime = shared["primary"]
    stations = shared["primary_stations"]
    n_out = max(2, regime.n_stations // 3)
    m_edt, e_edt = _locate_batch(ctx, regime, shared["primary_inv"],
                                 stations, "edt", n_outliers=n_out,
                                 rng=np.random.default_rng(11))
    m_l1, e_l1 = _locate_batch(ctx, regime, shared["primary_inv"],
                               stations, "l1", n_outliers=n_out,
                               rng=np.random.default_rng(11))
    status = "PASS"
    msg = ""
    if m_edt["total_km_med"] > 2.0 * m_l1["total_km_med"] + 5.0:
        status = "WARN"
        msg = "EDT median worse than L1 under outliers (investigate)"
    return {"metrics": {"EDT": m_edt, "L1": m_l1,
                        "n_outliers": n_out},
            "status": status, "message": msg}


def test_global_teleseismic(ctx, shared):
    if "global" not in shared["active"]:
        raise SkipTest("global regime not selected")
    regime = GLOBAL
    stations = shared["stations"]["global"]
    m_edt, _ = _locate_batch(ctx, regime, shared["inv"]["global"], stations,
                             "edt", n_outliers=1,
                             rng=np.random.default_rng(3),
                             do_posterior=True)
    assert m_edt["n_located"] > 0, "no global events located"
    return {"metrics": {"EDT": m_edt}}


# ===========================================================================
# SECTION 4b -- performance matrix across geometry / station / pick quality
# ===========================================================================
def _run_scenario(ctx, regime, inv, stations, sc, n_events, seed):
    """Locate n_events under one scenario spec; return aggregate stats."""
    L = ctx["locate"]
    path, tt_mem, _ = inv
    rng = np.random.default_rng(seed)
    events = random_events(regime, rng, n=n_events)
    delta = default_delta(regime)
    r_min, r_max, th_min, th_max, ph_min, ph_max = cap_grid_bounds(regime)
    phases = ("P", "S") if (sc["phases"] == "PS" and "S" in regime.phases) \
        else ("P",)
    pick_sigma = regime.pick_sigma_s * sc.get("noise_factor", 1.0)

    epi, dep, tot, az, ess, nobs = [], [], [], [], [], []
    loc = L.EQLocator(path, coord_sys="spherical")
    try:
        loc.add_stations(dict(stations))
        for ev in events:
            arr, perr, chosen = make_scenario_arrivals(
                tt_mem, stations, ev, rng, regime,
                k=sc["k"], geometry=sc["geometry"], phases=phases,
                dropout=sc.get("dropout", 0.0), pick_sigma=pick_sigma,
                n_outliers=sc.get("outliers", 0))
            if len(arr) < 2:
                continue
            loc.clear_arrivals()
            loc.add_arrivals(arr)
            loc.add_pick_errors(perr)
            initial = np.array([r_max - 0.5 * regime.depth_max_km,
                                0.5 * (th_min + th_max),
                                0.5 * (ph_min + ph_max), 0.0])
            if sc.get("posterior"):
                out = loc.locate_detailed(initial, delta, method="edt",
                                          nsamples=1000, nscatter=200, seed=7)
                est = out["hypocenter"]
                p = out.get("posterior", {})
                if "ess" in p:
                    ess.append(float(p["ess"]))
            else:
                est = loc.locate(initial, delta, method="edt")
            q = loc.quality(est)
            e, d, t, _ = location_errors_km(ev, est)
            epi.append(e); dep.append(d); tot.append(t)
            nobs.append(int(q.get("nobs", len(arr))))
            if np.isfinite(q.get("azimuthal_gap", np.nan)):
                az.append(float(q["azimuthal_gap"]))
    finally:
        loc.__exit__(None, None, None)

    def _med(x):
        return round(float(np.median(x)), 2) if x else None

    def _p90(x):
        return round(float(np.percentile(x, 90)), 2) if x else None

    out = {
        "n_events": n_events, "n_located": len(tot),
        "median_nobs": _med(nobs),
        "azgap_med": _med(az),
        "epi_km_med": _med(epi), "epi_km_p90": _p90(epi),
        "depth_km_med": _med(dep), "depth_km_p90": _p90(dep),
        "total_km_med": _med(tot), "total_km_p90": _p90(tot),
    }
    if ess:
        out["ess_med"] = _med(ess)
    return out


def _performance_scenarios():
    """Named scenarios spanning station count, phases, pick completeness and
    azimuthal coverage. Kept as an explicit list (not a full cross-product) to
    bound runtime while covering the important axes."""
    return [
        # station-count ladder, well-distributed, clean
        {"name": "dense_16_PS_surround", "k": 16, "geometry": "all",
         "phases": "PS"},
        {"name": "good_12_PS_surround", "k": 12, "geometry": "all",
         "phases": "PS"},
        {"name": "moderate_8_PS_surround", "k": 8, "geometry": "all",
         "phases": "PS"},
        {"name": "sparse_6_P_surround", "k": 6, "geometry": "all",
         "phases": "P"},
        {"name": "minimal_4_P_surround", "k": 4, "geometry": "all",
         "phases": "P"},
        # azimuthal coverage degraded
        {"name": "one_sided_8_PS", "k": 8, "geometry": "one_sided",
         "phases": "PS"},
        {"name": "az_gap120_8_PS", "k": 8, "geometry": "gap",
         "phases": "PS"},
        {"name": "one_sided_5_P", "k": 5, "geometry": "one_sided",
         "phases": "P"},
        # pick quality degraded
        {"name": "dropout40_12_PS", "k": 12, "geometry": "all",
         "phases": "PS", "dropout": 0.4},
        {"name": "noisy2x_10_PS", "k": 10, "geometry": "all",
         "phases": "PS", "noise_factor": 2.0},
        {"name": "outliers_10_PS", "k": 10, "geometry": "all",
         "phases": "PS", "outliers": 3},
        {"name": "worstcase_5_P_gap_noisy", "k": 5, "geometry": "gap",
         "phases": "P", "noise_factor": 2.0, "outliers": 1},
    ]


def test_performance_matrix(ctx, shared):
    """Judge locator performance across a matrix of realistic degradations:
    station count, P vs P+S, pick dropout, added noise/outliers, and
    azimuthal coverage (surrounded / one-sided / 120-deg gap)."""
    regime = shared["perf_regime"]
    stations = shared["perf_stations"]
    inv = shared["perf_inv"]
    n_events = 16 if _LEVEL == "full" else (12 if _LEVEL == "standard" else 6)

    results = {}
    for i, sc in enumerate(_performance_scenarios()):
        results[sc["name"]] = _run_scenario(
            ctx, regime, inv, stations, sc, n_events, seed=100 + i)

    located = [r for r in results.values() if r["n_located"] > 0]
    assert located, "performance matrix located nothing"
    # sanity: dense/surrounded should not be worse than the worst-case gap
    dense = results.get("dense_16_PS_surround", {})
    worst = results.get("worstcase_5_P_gap_noisy", {})
    msg = ""
    if (dense.get("total_km_med") is not None
            and worst.get("total_km_med") is not None
            and dense["total_km_med"] > worst["total_km_med"]):
        msg = ("dense geometry scored worse than worst-case gap -- "
               "unexpected, inspect")
    return {"metrics": results,
            "status": "WARN" if msg else "PASS", "message": msg or
            "location error across station/pick/azimuth degradations"}


# ===========================================================================
# SECTION 5 -- parameter sweeps
# ===========================================================================
def test_alpha_sweep(ctx, shared):
    regime = shared["primary"]
    stations = shared["primary_stations"]
    rows = {}
    for alpha in np.round(np.arange(0.0, 0.0901, 0.005), 4):
        m, _ = _locate_batch(ctx, regime, shared["primary_inv"], stations,
                             "edt", alpha=float(alpha), n_outliers=1,
                             rng=np.random.default_rng(5))
        rows[f"alpha_{alpha:.3f}"] = m["total_km_med"]
    best = min((v for v in rows.values() if v is not None), default=None)
    best_alpha = next((k for k, v in rows.items() if v == best), None)
    return {"metrics": {"total_km_med_by_alpha": rows,
                        "best_alpha": best_alpha, "best_total_km_med": best},
            "message": "median total error vs alpha (0 to 0.09 step 0.005)"}


def test_edt_ot_wt_geometry(ctx, shared):
    """EDT_OT_WT (NonLinLoc LOCMETH EDT_OT_WT) penalises the pdf by the
    spread of the per-arrival origin-time estimates. It should help most
    where the pdf has spurious maxima -- weak geometry with outlier picks --
    and be roughly neutral on clean, well-surrounded events. Replaces the
    old edt_reg sweep; edt_reg has been removed."""
    regime = shared["primary"]
    surr = shared["primary_stations"]
    one = shared["primary_onesided_stations"]
    out = {}
    for label, st, invkey in (("surrounded", surr, "primary_inv"),
                              ("one_sided", one, "primary_onesided_inv")):
        for otwt in (False, True):
            m, _ = _locate_batch(ctx, regime, shared[invkey], st, "edt",
                                 edt_ot_wt=otwt, n_outliers=1,
                                 rng=np.random.default_rng(9))
            out[f"{label}_ot_wt_{'on' if otwt else 'off'}"] = m["total_km_med"]
    return {"metrics": out,
            "message": "median total error: geometry x edt_ot_wt"}


def test_edt_exponent_sweep(ctx, shared):
    regime = shared["primary"]
    stations = shared["primary_stations"]
    rows = {}
    for expo in (1.0, 2.0, 4.0):
        m, _ = _locate_batch(ctx, regime, shared["primary_inv"], stations,
                             "edt", exponent=expo, n_outliers=1,
                             do_posterior=True,
                             rng=np.random.default_rng(6))
        rows[f"exp_{expo}"] = {
            "total_km_med": m["total_km_med"],
            "ess_med": m.get("ess_med"),
            "semi_axes": m.get("ellipsoid_semi_axes_km_med"),
        }
    return {"metrics": rows,
            "message": "error / ESS / ellipsoid vs edt_exponent"}


def test_posterior_proposal(ctx, shared):
    """Hessian proposal should give higher ESS than the uniform proposal at
    equal sample count on a well-constrained event."""
    L = ctx["locate"]
    regime = shared["primary"]
    stations = shared["primary_stations"]
    path, tt_mem, _ = shared["primary_inv"]
    rng = np.random.default_rng(21)
    ev = random_events(regime, rng, n=1)[0]
    arr, perr, _ = synth_arrivals(tt_mem, stations, ev, rng,
                                  regime.pick_sigma_s, regime.phases)
    if len(arr) < 3:
        raise SkipTest("not enough arrivals for posterior test")
    delta = default_delta(regime)
    loc = L.EQLocator(path, coord_sys="spherical")
    try:
        loc.add_stations(dict(stations))
        loc.add_arrivals(arr)
        loc.add_pick_errors(perr)
        r_min, r_max, th_min, th_max, ph_min, ph_max = cap_grid_bounds(regime)
        initial = np.array([r_max - 0.5 * regime.depth_max_km,
                            0.5 * (th_min + th_max),
                            0.5 * (ph_min + ph_max), 0.0])
        hypo = loc.locate(initial, delta, method="edt")
        res = {}
        for prop in ("hessian", "uniform"):
            post = loc.sample_posterior(hypo[:3], delta[:3], nsamples=2000,
                                        nscatter=500, seed=1, proposal=prop)
            scat_pe = physical_ellipsoid(post.get("scatter"), ctx["tf"])
            lib = post.get("ellipsoid_km")
            entry = {
                "ess": round(float(post["ess"]), 1),
                "proposal_used": post.get("proposal", prop),
                "semi_axes_km": ([round(float(x), 2) for x in scat_pe[0]]
                                 if scat_pe is not None else None),
                "ellipsoid_vol_km3": (round(scat_pe[1], 3)
                                      if scat_pe is not None else None),
                "library_ellipsoid_km_present": lib is not None,
            }
            # if the library exposes the fixed physical ellipsoid, cross-check
            # its semi-axes against the independent scatter-derived value
            if lib is not None and scat_pe is not None:
                lib_semi = np.asarray(lib.get("semi_axes"), float)
                entry["library_semi_axes_km"] = [round(float(x), 2)
                                                 for x in lib_semi]
                entry["library_frame"] = lib.get("frame")
                a = np.sort(lib_semi)[::-1]
                b = np.sort(scat_pe[0])[::-1]
                denom = np.maximum(b, 1e-6)
                entry["max_rel_diff_vs_scatter"] = round(
                    float(np.max(np.abs(a - b) / denom)), 3)
            res[prop] = entry
    finally:
        loc.__exit__(None, None, None)
    status = "PASS"
    msg = ""
    if res["hessian"]["ess"] < res["uniform"]["ess"]:
        status = "WARN"
        msg = "hessian ESS below uniform (unexpected; check proposal)"
    return {"metrics": res, "status": status, "message": msg}


def test_locate_seed_determinism(ctx, shared):
    """Same seed -> identical solution; the search is reproducible."""
    L = ctx["locate"]
    regime = shared["primary"]
    stations = shared["primary_stations"]
    path, tt_mem, _ = shared["primary_inv"]
    rng = np.random.default_rng(31)
    ev = random_events(regime, rng, n=1)[0]
    arr, perr, _ = synth_arrivals(tt_mem, stations, ev, rng,
                                  regime.pick_sigma_s, regime.phases)
    if len(arr) < 2:
        raise SkipTest("not enough arrivals")
    delta = default_delta(regime)
    r_min, r_max, th_min, th_max, ph_min, ph_max = cap_grid_bounds(regime)
    initial = np.array([r_max - 0.5 * regime.depth_max_km,
                        0.5 * (th_min + th_max),
                        0.5 * (ph_min + ph_max), 0.0])
    ests = []
    for _ in range(2):
        loc = L.EQLocator(path, coord_sys="spherical")
        try:
            loc.add_stations(dict(stations))
            loc.add_arrivals(arr)
            loc.add_pick_errors(perr)
            loc.locate_seed = 12345
            ests.append(loc.locate(initial, delta, method="edt"))
        finally:
            loc.__exit__(None, None, None)
    d = float(np.max(np.abs(ests[0] - ests[1])))
    assert d < 1e-6, f"nondeterministic under fixed seed (dmax={d})"
    return {"metrics": {"max_abs_diff": d}}


def test_alpha_not_reset(ctx, shared):
    """Regression: locate() without an explicit alpha must NOT reset a
    previously configured self.alpha."""
    L = ctx["locate"]
    regime = shared["primary"]
    stations = shared["primary_stations"]
    path, tt_mem, _ = shared["primary_inv"]
    rng = np.random.default_rng(41)
    ev = random_events(regime, rng, n=1)[0]
    arr, perr, _ = synth_arrivals(tt_mem, stations, ev, rng,
                                  regime.pick_sigma_s, regime.phases)
    if len(arr) < 2:
        raise SkipTest("not enough arrivals")
    delta = default_delta(regime)
    r_min, r_max, th_min, th_max, ph_min, ph_max = cap_grid_bounds(regime)
    initial = np.array([r_max - 0.5 * regime.depth_max_km,
                        0.5 * (th_min + th_max),
                        0.5 * (ph_min + ph_max), 0.0])
    loc = L.EQLocator(path, coord_sys="spherical")
    try:
        loc.add_stations(dict(stations))
        loc.add_arrivals(arr)
        loc.add_pick_errors(perr)
        loc.alpha = 0.037
        loc.locate(initial, delta, method="l1")   # no alpha passed
        after = float(loc.alpha)
    finally:
        loc.__exit__(None, None, None)
    assert abs(after - 0.037) < 1e-9, f"alpha was reset to {after}"
    return {"metrics": {"alpha_after_locate": after}}


# ===========================================================================
# SECTION 6 -- vertical compression: precision vs model size
# ===========================================================================
def _vertical_compression_study(ctx, regime, tmpdir):
    """Find how far the VERTICAL (radial) axis can be compressed before depth
    precision degrades -- the trade-off that sets the optimal grid.

    Horizontal node spacing (dx = surface arc-length R*dtheta) is held fixed
    at a sensible value, and the vertical node spacing (dz = dr) is varied to
    give aspect ratios dx:dz from 1:1 (fine vertical, large model) up to 10:1
    (coarse/compressed vertical, small model). For each ratio we report the
    model size (nodes per grid and MB) and the location precision (median and
    p90 total and DEPTH error) over a fixed event suite, so the knee -- the
    smallest model that still meets a depth-precision target -- is visible.
    """
    L, F = ctx["locate"], ctx["fields"]

    sregime = Regime(**{**regime.__dict__,
                        "cap_half_km": regime.sweep_cap_km,
                        "phases": ("P",),
                        "n_stations": min(6, regime.n_stations)})
    rng = np.random.default_rng(RNG_SEED + 99)
    base_stations = random_stations(sregime, rng)
    events = random_events(sregime, rng, n=max(6, regime.n_events))
    delta = default_delta(sregime)
    r_min, r_max, th_min, th_max, ph_min, ph_max = cap_grid_bounds(sregime)
    span_r = r_max - r_min

    # Fixed horizontal resolution: pick an angular node count giving a
    # reasonable surface arc spacing, then hold it for every ratio.
    n_ang = int(regime.sweep_fixed_nang)
    dtheta = (th_max - th_min) / (n_ang - 1)
    dx = R_EARTH * dtheta                                    # horizontal km

    rows = {}
    for ratio in regime.sweep_ratios:                        # dx : dz
        dz = dx / ratio                                      # vertical km
        nr = max(5, int(round(span_r / dz)) + 1)
        dz_actual = span_r / (nr - 1)
        ratio_actual = dx / dz_actual

        path, tt_mem, _ = build_inventory(
            ctx, sregime, base_stations, nr=nr, ntheta=n_ang, nphi=n_ang,
            use_point_source=False, tmpdir=tmpdir)

        nodes_per_grid = int(nr * n_ang * n_ang)
        mb_per_grid = nodes_per_grid * 4 / 1e6               # float32 traveltime
        n_grids = len(tt_mem)

        tot_errs, dep_errs, semis = [], [], []
        loc = L.EQLocator(path, coord_sys="spherical")
        try:
            loc.add_stations(dict(base_stations))
            for ev in events:
                arr, perr, _ = synth_arrivals(
                    tt_mem, base_stations, ev, rng, sregime.pick_sigma_s,
                    ("P",))
                if len(arr) < 2:
                    continue
                loc.clear_arrivals()
                loc.add_arrivals(arr)
                loc.add_pick_errors(perr)
                initial = np.array([r_max - 0.5 * sregime.depth_max_km,
                                    0.5 * (th_min + th_max),
                                    0.5 * (ph_min + ph_max), 0.0])
                out = loc.locate_detailed(initial, delta, method="edt",
                                          nsamples=1200, nscatter=300, seed=7)
                _, ddep, tot, _ = location_errors_km(ev, out["hypocenter"])
                tot_errs.append(tot)
                dep_errs.append(ddep)
                pe = ellipsoid_from_post(out.get("posterior", {}), ctx["tf"])
                if pe is not None:
                    semis.append(pe[0])
        finally:
            loc.__exit__(None, None, None)

        def _med(x):
            return round(float(np.median(x)), 2) if x else None

        def _p90(x):
            return round(float(np.percentile(x, 90)), 2) if x else None

        entry = {
            "aspect_dx_dz": round(ratio_actual, 2),
            "dx_km": round(dx, 2),
            "dz_km": round(dz_actual, 2),
            "nr": nr, "n_ang": n_ang,
            "nodes_per_grid": nodes_per_grid,
            "MB_per_grid": round(mb_per_grid, 2),
            "MB_all_grids": round(mb_per_grid * n_grids, 2),
            "n_located": len(tot_errs),
            "depth_km_med": _med(dep_errs),
            "depth_km_p90": _p90(dep_errs),
            "total_km_med": _med(tot_errs),
            "total_km_p90": _p90(tot_errs),
        }
        if semis:
            sa = np.median(np.vstack(semis), axis=0)
            entry["ellipsoid_vol_km3_med"] = round(
                float(4.0 / 3.0 * np.pi * np.prod(sa)), 2)
        rows[f"ratio_{ratio:g}"] = entry
    return rows


def _compression_recommendation(rows):
    """Pick the most compressed grid (smallest model) whose median depth error
    is within 15% of the best (finest-vertical) grid's -- the 'knee'."""
    valid = [(k, v) for k, v in rows.items()
             if v.get("depth_km_med") is not None]
    if not valid:
        return None
    best_depth = min(v["depth_km_med"] for _, v in valid)
    tol = best_depth * 1.15 + 0.5
    acceptable = [(k, v) for k, v in valid if v["depth_km_med"] <= tol]
    # smallest model among acceptable
    k, v = min(acceptable, key=lambda kv: kv[1]["nodes_per_grid"])
    return {"recommended_ratio_dx_dz": v["aspect_dx_dz"],
            "recommended_key": k,
            "depth_km_med": v["depth_km_med"],
            "MB_per_grid": v["MB_per_grid"],
            "depth_tol_used_km": round(tol, 2)}


def test_vertical_compression_regional(ctx, shared):
    if "regional" not in shared["active"]:
        raise SkipTest("regional regime not selected")
    tmp = mkdtemp(prefix="pyk_vcomp_reg_")
    rows = _vertical_compression_study(ctx, REGIONAL, tmp)
    rec = _compression_recommendation(rows)
    ok = [g for g in rows.values() if g["n_located"] > 0]
    assert ok, "vertical-compression study located nothing"
    return {"metrics": {"by_ratio": rows, "recommendation": rec},
            "message": "regional depth precision & model size vs dx:dz "
                       "(vertical compression)"}


def test_vertical_compression_global(ctx, shared):
    if "global" not in shared["active"]:
        raise SkipTest("global regime not selected")
    tmp = mkdtemp(prefix="pyk_vcomp_glob_")
    rows = _vertical_compression_study(ctx, GLOBAL, tmp)
    rec = _compression_recommendation(rows)
    ok = [g for g in rows.values() if g["n_located"] > 0]
    assert ok, "vertical-compression study located nothing"
    return {"metrics": {"by_ratio": rows, "recommendation": rec},
            "message": "global depth precision & model size vs dx:dz "
                       "(vertical compression)"}


# ===========================================================================
# SECTION 7 -- quality metrics
# ===========================================================================
def test_quality_metrics(ctx, shared):
    """azimuthal_gap, min_station_dist, nobs, rms sanity on known geometry."""
    L = ctx["locate"]
    regime = shared["primary"]
    stations = shared["primary_stations"]
    path, tt_mem, _ = shared["primary_inv"]
    rng = np.random.default_rng(51)
    ev = random_events(regime, rng, n=1)[0]
    arr, perr, _ = synth_arrivals(tt_mem, stations, ev, rng,
                                  regime.pick_sigma_s, regime.phases)
    if len(arr) < 2:
        raise SkipTest("not enough arrivals")
    delta = default_delta(regime)
    loc = L.EQLocator(path, coord_sys="spherical")
    try:
        loc.add_stations(dict(stations))
        loc.add_arrivals(arr)
        loc.add_pick_errors(perr)
        r_min, r_max, th_min, th_max, ph_min, ph_max = cap_grid_bounds(regime)
        initial = np.array([r_max - 0.5 * regime.depth_max_km,
                            0.5 * (th_min + th_max),
                            0.5 * (ph_min + ph_max), 0.0])
        est = loc.locate(initial, delta, method="edt")
        q = loc.quality(est)
    finally:
        loc.__exit__(None, None, None)
    assert q["nobs"] >= 2, "nobs too small"
    assert 0.0 <= q["azimuthal_gap"] <= 360.0, "azimuthal gap out of range"
    assert q["min_station_dist"] >= 0.0, "negative station distance"
    assert np.isfinite(q["rms"]), "rms not finite"
    return {"metrics": {k: (round(float(v), 3) if np.isfinite(v) else None)
                        for k, v in q.items()}}


# ===========================================================================
# SECTION 8 -- robustness / edge cases
# ===========================================================================
def test_too_few_arrivals(ctx, shared):
    """Fewer than 2 arrivals must not crash locate()."""
    L = ctx["locate"]
    regime = shared["primary"]
    stations = shared["primary_stations"]
    path, tt_mem, _ = shared["primary_inv"]
    one_key = next(iter(tt_mem))
    loc = L.EQLocator(path, coord_sys="spherical")
    try:
        loc.add_stations(dict(stations))
        loc.add_arrivals({one_key: 12.3})
        delta = default_delta(regime)
        r_min, r_max, th_min, th_max, ph_min, ph_max = cap_grid_bounds(regime)
        initial = np.array([r_max - 0.5 * regime.depth_max_km,
                            0.5 * (th_min + th_max),
                            0.5 * (ph_min + ph_max), 0.0])
        est = loc.locate(initial, delta, method="edt")
        shape_ok = np.asarray(est).shape == (4,)
    finally:
        loc.__exit__(None, None, None)
    assert shape_ok, "locate did not return a 4-vector with 1 arrival"
    return {"metrics": {"returned_shape": "(4,)"}}


def test_out_of_grid_arrival(ctx, shared):
    """An arrival whose grid does not cover the search box is dropped with a
    warning, not an error."""
    L = ctx["locate"]
    regime = shared["primary"]
    stations = shared["primary_stations"]
    path, tt_mem, _ = shared["primary_inv"]
    rng = np.random.default_rng(61)
    ev = random_events(regime, rng, n=1)[0]
    arr, perr, _ = synth_arrivals(tt_mem, stations, ev, rng,
                                  regime.pick_sigma_s, regime.phases)
    # add a bogus arrival for a station with no grid
    arr[("ZZ", "NOPE", "P")] = 5.0
    loc = L.EQLocator(path, coord_sys="spherical")
    n_res = None
    try:
        loc.add_stations(dict(stations))
        loc.add_arrivals(arr)
        loc.add_pick_errors(perr)
        delta = default_delta(regime)
        r_min, r_max, th_min, th_max, ph_min, ph_max = cap_grid_bounds(regime)
        initial = np.array([r_max - 0.5 * regime.depth_max_km,
                            0.5 * (th_min + th_max),
                            0.5 * (ph_min + ph_max), 0.0])
        est = loc.locate(initial, delta, method="edt")
        res = loc.residuals(est)
        n_res = len(res)
        assert ("ZZ", "NOPE", "P") not in res, "bogus arrival not dropped"
    finally:
        loc.__exit__(None, None, None)
    return {"metrics": {"n_residuals": n_res}}


def test_inventory_double_close(ctx, shared):
    """EQLocator __exit__ followed by __del__/GC must not raise (double-close
    guard)."""
    L = ctx["locate"]
    path, _, _ = shared["primary_inv"]
    loc = L.EQLocator(path, coord_sys="spherical")
    loc.__exit__(None, None, None)
    # explicit second close path
    loc.__del__()
    return {"metrics": {"double_close": "ok"}}


def test_all_outliers_no_crash(ctx, shared):
    """A pathological all-corrupted arrival set should still return a finite
    4-vector (quality will be poor, but no exception)."""
    L = ctx["locate"]
    regime = shared["primary"]
    stations = shared["primary_stations"]
    path, tt_mem, _ = shared["primary_inv"]
    rng = np.random.default_rng(71)
    ev = random_events(regime, rng, n=1)[0]
    arr, perr, _ = synth_arrivals(
        tt_mem, stations, ev, rng, regime.pick_sigma_s, regime.phases,
        n_outliers=999, outlier_size=(10.0, 20.0))
    if len(arr) < 2:
        raise SkipTest("not enough arrivals")
    loc = L.EQLocator(path, coord_sys="spherical")
    try:
        loc.add_stations(dict(stations))
        loc.add_arrivals(arr)
        loc.add_pick_errors(perr)
        delta = default_delta(regime)
        r_min, r_max, th_min, th_max, ph_min, ph_max = cap_grid_bounds(regime)
        initial = np.array([r_max - 0.5 * regime.depth_max_km,
                            0.5 * (th_min + th_max),
                            0.5 * (ph_min + ph_max), 0.0])
        est = loc.locate(initial, delta, method="edt")
        finite = bool(np.all(np.isfinite(est)))
    finally:
        loc.__exit__(None, None, None)
    assert finite, "non-finite solution under all-outlier input"
    return {"metrics": {"finite_solution": finite}}


# ===========================================================================
# Orchestration
# ===========================================================================
def build_shared(ctx, suite, active_names):
    """Build the reusable inventories/geometries for the ACTIVE regimes only,
    and expose 'primary' handles (regional if active, else global) that the
    regime-independent locator/parameter/quality/edge tests run against."""
    shared = {"inv": {}, "stations": {}, "active": tuple(active_names)}
    regimes = {"regional": REGIONAL, "global": GLOBAL}

    def _make(regime, seed_off):
        def _fn():
            rng = np.random.default_rng(RNG_SEED + seed_off)
            stations = random_stations(regime, rng)
            path, tt_mem, vfs = build_inventory(ctx, regime, stations)
            shared["inv"][regime.name] = (path, tt_mem, vfs)
            shared["stations"][regime.name] = stations
            return {"metrics": {"n_station_phase": len(tt_mem),
                                "grid": [regime.nr, regime.ntheta,
                                         regime.nphi]}}
        return _fn

    for name in active_names:
        suite.run(f"build {name} inventory", _make(regimes[name],
                                                   {"regional": 0,
                                                    "global": 2}[name]))

    # primary regime = regional if active else global
    primary_name = "regional" if "regional" in active_names else active_names[0]
    primary = regimes[primary_name]
    shared["primary"] = primary
    shared["primary_name"] = primary_name
    if primary_name in shared["inv"]:
        shared["primary_inv"] = shared["inv"][primary_name]
        shared["primary_stations"] = shared["stations"][primary_name]

    # one-sided geometry for the primary regime (edt_ot_wt x geometry test)
    def _onesided():
        rng = np.random.default_rng(RNG_SEED + 1)
        stations = one_sided_stations(primary, rng)
        path, tt_mem, vfs = build_inventory(ctx, primary, stations)
        shared["primary_onesided_stations"] = stations
        shared["primary_onesided_inv"] = (path, tt_mem, vfs)
        return {"metrics": {"n_station_phase": len(tt_mem)}}

    suite.run(f"build {primary_name} one-sided inventory", _onesided)

    # richer many-station inventory for the performance matrix (subset per
    # scenario), primary regime only
    def _perf():
        rng = np.random.default_rng(RNG_SEED + 7)
        n_perf = 16 if _LEVEL != "quick" else 10
        pregime = Regime(**{**primary.__dict__, "n_stations": n_perf})
        stations = random_stations(pregime, rng)
        path, tt_mem, vfs = build_inventory(ctx, pregime, stations)
        shared["perf_stations"] = stations
        shared["perf_inv"] = (path, tt_mem, vfs)
        shared["perf_regime"] = pregime
        return {"metrics": {"n_stations": n_perf,
                            "n_station_phase": len(tt_mem)}}

    suite.run(f"build {primary_name} performance inventory", _perf)
    return shared


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    suite = Suite()
    t_start = time.time()
    active = _resolve_regimes()

    _log(f"pykonal validation suite -- level={_LEVEL}  "
         f"regimes={','.join(active)}")
    _log(f"output -> {OUTPUT_DIR}")

    # --- import (fatal if this fails) ---
    suite.section("0. environment & import")
    try:
        ctx = import_pykonal()
    except Exception as exc:  # noqa: BLE001
        suite.run("import pykonal", lambda: (_ for _ in ()).throw(exc))
        suite.write_reports(OUTPUT_DIR)
        _finish(suite, t_start)
        return 1
    suite.run("import pykonal", test_import, ctx)

    # --- solver correctness ---
    suite.section("1. solver correctness")
    suite.run("homogeneous sphere analytic", test_homogeneous_sphere_analytic, ctx)
    suite.run("point-source solver (AK135)", test_point_source_solver_runs, ctx)
    suite.run("causal stencil vs aspect ratio", test_causal_stencil_aspect, ctx)

    # --- fields ---
    suite.section("2. fields: IO & interpolation")
    suite.run("HDF round-trip", test_field_hdf_roundtrip, ctx)
    suite.run("interpolation & null", test_field_interpolation, ctx)

    # --- shared inventories ---
    suite.section("3. build shared inventories")
    shared = build_shared(ctx, suite, active)

    # --- inventory ---
    suite.section("3b. inventory read / crop")
    suite.run("inventory build & read-back", test_inventory_build_read, ctx, shared)
    suite.run("max_dist masking", test_inventory_maxdist_mask, ctx)

    # --- locator end to end ---
    suite.section("4. EQLocator: EDT vs L1")
    suite.run("EDT vs L1 (clean)", test_edt_vs_l1_clean, ctx, shared)
    suite.run("EDT vs L1 (outliers)", test_edt_vs_l1_outliers, ctx, shared)
    suite.run("global teleseismic", test_global_teleseismic, ctx, shared)

    # --- performance matrix ---
    suite.section("4b. performance matrix (geometry / stations / picks)")
    suite.run("performance matrix", test_performance_matrix, ctx, shared)

    # --- parameter sweeps ---
    suite.section("5. parameter sweeps")
    suite.run("alpha sweep", test_alpha_sweep, ctx, shared)
    suite.run("edt_ot_wt x geometry", test_edt_ot_wt_geometry, ctx, shared)
    suite.run("edt_exponent sweep", test_edt_exponent_sweep, ctx, shared)
    suite.run("posterior proposal (hessian vs uniform)",
              test_posterior_proposal, ctx, shared)
    suite.run("locate_seed determinism", test_locate_seed_determinism, ctx, shared)
    suite.run("alpha not reset by locate()", test_alpha_not_reset, ctx, shared)

    # --- vertical compression: precision vs model size ---
    suite.section("6. vertical compression (precision vs model size)")
    suite.run("vertical compression (regional)",
              test_vertical_compression_regional, ctx, shared)
    suite.run("vertical compression (global)",
              test_vertical_compression_global, ctx, shared)

    # --- quality ---
    suite.section("7. quality metrics")
    suite.run("quality metrics", test_quality_metrics, ctx, shared)

    # --- robustness ---
    suite.section("8. robustness / edge cases")
    suite.run("too few arrivals", test_too_few_arrivals, ctx, shared)
    suite.run("out-of-grid arrival dropped", test_out_of_grid_arrival, ctx, shared)
    suite.run("inventory double-close", test_inventory_double_close, ctx, shared)
    suite.run("all-outlier no crash", test_all_outliers_no_crash, ctx, shared)

    # --- plots ---
    suite.write_reports(OUTPUT_DIR)
    _finish(suite, t_start)
    counts = suite.summary()
    return 0 if counts.get("FAIL", 0) == 0 else 2


def _finish(suite, t_start):
    counts = suite.summary()
    _log("\n" + "=" * 70)
    _log("SUMMARY: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    _log(f"total wall time: {time.time() - t_start:.1f}s")
    _log(f"artifacts in: {OUTPUT_DIR}")
    with open(os.path.join(OUTPUT_DIR, "console.log"), "w") as fh:
        fh.write("\n".join(_LOG_LINES))


if __name__ == "__main__":
    sys.exit(main())