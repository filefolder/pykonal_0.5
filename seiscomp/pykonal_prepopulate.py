#!/usr/bin/env python3
"""
Pre-populate pykonal traveltime grids for every station in the configured
station CSV, so that scolv relocations never have to wait for an FMM solve.

Usage:
    pykonal_prepopulate.py /path/to/pykonal_locext.json

Run it with the SAME interpreter the LocExt profile uses (it imports
pykonal_locext to reuse the exact coordinate transform, storage layout and
model-fingerprinting logic, so the grids it writes are the ones the plugin
will find).

Stations whose coordinates fall outside the velocity model are skipped and
listed - they cannot have a grid computed, and the plugin excludes their
arrivals at location time anyway. Grids that already exist and were built
from the current velocity model are left alone; grids built from a
superseded model are rebuilt.
"""
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pykonal_locext as plx          # noqa: E402
from pykonal import fields            # noqa: E402


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    cfg_path = sys.argv[1]
    with open(cfg_path) as f:
        cfg = json.load(f)

    velocity_models = cfg.get("velocity_models") or {}
    if not velocity_models:
        print("config has no 'velocity_models': nothing can be built.",
              file=sys.stderr)
        return 1
    stations_csv = cfg.get("stations_csv")
    if not stations_csv:
        print("config has no 'stations_csv': no station list to build from.",
              file=sys.stderr)
        return 1

    transform = plx.GeoTransform(cfg)

    # ---------------- stations
    stations = {}
    with open(stations_csv) as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 4:
                continue
            net, sta = row[0].strip(), row[1].strip()
            if net.lower() == 'network': continue
            try:
                lat, lon = float(row[2]), float(row[3])
                elev = float(row[4]) if len(row) > 4 and row[4].strip() else 0.0
            except ValueError:
                continue           # header line or malformed row
            stations[(net, sta)] = (lat, lon, elev)
    print(f"stations in CSV                 : {len(stations)}")

    # ---------------- velocity model extents, per phase
    extents = {}
    for phase, vm in velocity_models.items():
        try:
            f = fields.read_hdf(vm) if isinstance(vm, str) else vm
            minc = np.asarray(f.min_coords, dtype=float)
            npts = np.asarray(f.npts)
            node = np.asarray(f.node_intervals, dtype=float)
            extents[phase.upper()] = (minc, minc + (npts - 1) * node)
        except Exception as err:
            print(f"could not read velocity model for phase {phase}: {err}",
                  file=sys.stderr)
    if not extents:
        return 1

    # ---------------- build the request set, skipping out-of-model stations
    requests = {}
    skipped = {}
    clamped = set()
    vertical_axis = 2 if transform.spatial_axes_are_xyz_enu() else 0
    for (net, sta), (lat, lon, elev) in sorted(stations.items()):
        grid_xyz = transform.geo_to_grid(lat, lon, -elev / 1000.0)
        for phase in sorted(extents):
            lo, hi = extents[phase]
            fitted = plx.clamp_to_grid(grid_xyz, lo, hi, vertical_axis)
            if fitted is None:
                # laterally outside the model: no grid can be computed
                skipped.setdefault(f"{net}.{sta}", []).append(phase)
                continue
            # absolute comparison: allclose's relative tolerance on r~6371
            # would hide clamps of tens of metres
            if np.any(np.abs(fitted - np.asarray(grid_xyz, dtype=float)) > 1e-9):
                clamped.add(f"{net}.{sta}")
            requests[(net, sta, phase)] = fitted

    print(f"phases                          : {', '.join(sorted(extents))}")
    print(f"station/phase grids requested    : {len(requests)}")
    if clamped:
        print(f"elevation clamped onto the grid : {len(clamped)} station(s) "
              f"(above the model top; placed at the shallowest node)")
    if skipped:
        print(f"outside the velocity model      : {len(skipped)} station(s) "
              f"- skipped, no grid possible")
        for name in sorted(skipped):
            print(f"    {name} ({', '.join(skipped[name])})")
    if not requests:
        print("nothing to do.")
        return 0

    # ---------------- build
    max_dist = float(cfg.get("max_dist_km", 900.0))
    nproc = cfg.get("ensure_nproc")
    shard_dir = cfg.get("traveltime_dir")
    t0 = time.time()

    if shard_dir:
        print(f"storage                         : one file per station-phase "
              f"in {shard_dir}")
        report = plx.ensure_traveltimes_sharded(
            shard_dir, requests, velocity_models,
            max_dist=max_dist, nproc=nproc,
        )
    else:
        inv_path = cfg["traveltime_inventory"]
        print(f"storage                         : single inventory {inv_path}")
        from pykonal.inventory import ensure_traveltimes
        with plx.inventory_lock(inv_path, shared=False):
            report = ensure_traveltimes(
                inv_path, requests, velocity_models,
                max_dist=max_dist, nproc=nproc,
            )

    dt = time.time() - t0
    built = report.get("computed", [])
    present = report.get("present", [])
    failed = report.get("skipped", [])
    print()
    print(f"already current                 : {len(present)}")
    print(f"built now                       : {len(built)}")
    print(f"could not build                 : {len(failed)}")
    for k in failed:
        print(f"    {'.'.join(k)}")
    print(f"elapsed                         : {dt:.1f} s"
          + (f"  ({dt/len(built):.1f} s per grid)" if built else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
