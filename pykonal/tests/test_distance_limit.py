"""
Test distance-limited traveltime grids (default 900 km station-epicenter).

Scenario: continental-scale cartesian grid (2400 x 2400 x 60 km), sparse
network. Compare full vs 900-km-limited inventories on storage and on
location results for (a) an event inside every station's radius and
(b) an event beyond 900 km from some stations.
"""
import os
import numpy as np
import pykonal
from pykonal.inventory import TraveltimeInventory
from pykonal.locate import EQLocator

rng = np.random.default_rng(3)

min_coords = (0.0, 0.0, 0.0)
node_intervals = (6.0, 6.0, 3.0)
npts = (401, 401, 21)          # 2400 x 2400 x 60 km
zz = np.arange(npts[2]) * node_intervals[2]
velocity = np.broadcast_to(6.0 + 0.02 * zz, npts).copy()

stations = {
    ("XX", "S01"): (600.0, 600.0, 0.0),
    ("XX", "S02"): (1800.0, 600.0, 0.0),
    ("XX", "S03"): (600.0, 1800.0, 0.0),
    ("XX", "S04"): (1800.0, 1800.0, 0.0),
    ("XX", "S05"): (1200.0, 1200.0, 0.0),
    ("XX", "S06"): (900.0, 1500.0, 0.0),
}

print("solving FMM grids (6 stations, 3.4M nodes each)...")
tt_fields = {}
for (net, sta), coords in stations.items():
    solver = pykonal.solver.PointSourceSolver(coord_sys="cartesian")
    solver.velocity.min_coords = min_coords
    solver.velocity.node_intervals = node_intervals
    solver.velocity.npts = npts
    solver.velocity.values = velocity
    solver.src_loc = np.array(coords)
    solver.solve()
    tt_fields[(net, sta, "P")] = solver.traveltime

full_path = "./data/tt_full.h5"
lim_path = "./data/tt_900km.h5"
for p in (full_path, lim_path):
    if os.path.exists(p):
        os.remove(p)

inv = TraveltimeInventory(full_path, mode="w")
for key, f in tt_fields.items():
    inv.add(f, "/".join(key), max_dist=None, compress=False)  # legacy behavior
inv.f5.close()

inv = TraveltimeInventory(lim_path, mode="w")
for key, f in tt_fields.items():
    inv.add(f, "/".join(key), station_coords=np.array(stations[key[:2]]))  # max_dist=900 default
inv.f5.close()

s_full = os.path.getsize(full_path) / 1e6
s_lim = os.path.getsize(lim_path) / 1e6
print(f"\nstorage: full = {s_full:.1f} MB   900-km limited = {s_lim:.1f} MB   "
      f"({s_full / s_lim:.1f}x smaller)")

# --------------------------------------------------------------- events
def make_event(hypo, t0, keys):
    return {k: t0 + tt_fields[k].value(np.array(hypo)) + rng.normal(0, 0.05)
            for k in keys}

initial_offset = np.array([40.0, -30.0, 3.0, 1.0])
delta = np.array([150.0, 150.0, 25.0, 20.0])
all_keys = list(tt_fields)

def locate_both(hypo, t0):
    arrivals = make_event(hypo, t0, all_keys)
    out = {}
    for tag, path in (("full", full_path), ("900km", lim_path)):
        with EQLocator(path, coord_sys="cartesian") as loc:
            loc.add_arrivals(arrivals)
            initial = np.array([*hypo, t0]) + initial_offset
            soln = loc.locate(initial, delta.copy(), 0.02, "edt")
            nused = len(loc.residuals(soln))
            out[tag] = (soln, nused)
    return out

print("\nEvent A: central (within 900 km of all 6 stations)")
true_a = (1200.0, 1200.0, 12.0)
res = locate_both(true_a, 500.0)
for tag, (s, n) in res.items():
    err = np.linalg.norm(s[:3] - true_a)
    print(f"  {tag:6s}: loc err = {err:5.2f} km  t0 err = {s[3]-500:+.3f} s  "
          f"picks used = {n}/6")
d = np.linalg.norm(res["full"][0][:3] - res["900km"][0][:3])
print(f"  full-vs-limited solution difference: {d:.2f} km")

print("\nEvent B: NE quadrant (3 stations beyond the 900 km cutoff)")
true_b = (1700.0, 1600.0, 15.0)
for k in all_keys:
    dist = np.linalg.norm(np.array(stations[k[:2]][:2]) - np.array(true_b[:2]))
    print(f"    {k[1]}: {dist:6.0f} km {'(beyond cutoff)' if dist > 900 else ''}")
res = locate_both(true_b, 500.0)
for tag, (s, n) in res.items():
    err = np.linalg.norm(s[:3] - true_b)
    print(f"  {tag:6s}: loc err = {err:5.2f} km  t0 err = {s[3]-500:+.3f} s  "
          f"picks used = {n}/6")

print("\nEvent C: SW quadrant, arrivals supplied from ALL stations")
print("  (locator must drop uncovered arrivals, not crash or bias)")
true_c = (700.0, 800.0, 10.0)
arrivals = make_event(true_c, 500.0, all_keys)
with EQLocator(lim_path, coord_sys="cartesian") as loc:
    loc.add_arrivals(arrivals)
    soln = loc.locate(np.array([*true_c, 500.0]) + initial_offset,
                      delta.copy(), 0.02, "edt")
    used = loc.residuals(soln)
    err = np.linalg.norm(soln[:3] - true_c)
    print(f"  900km : loc err = {err:5.2f} km  t0 err = {soln[3]-500:+.3f} s  "
          f"picks used = {len(used)}/6 -> {sorted(k[1] for k in used)}")

# --------------------------------------------------------------- spherical
print("\nSpherical-grid crop check (35 x 35 deg regional grid, 1 station)")
from pykonal.transformations import geo2sph
solver = pykonal.solver.PointSourceSolver(coord_sys="spherical")
solver.velocity.min_coords = geo2sph(np.array([70.0, 15.0, 100.0]))  # min colat = 70N
solver.velocity.node_intervals = (5.0, np.radians(0.1), np.radians(0.1))
solver.velocity.npts = (21, 351, 351)
solver.velocity.values = np.full((21, 351, 351), 8.0)
solver.src_loc = geo2sph(np.array([52.5, 32.5, 0.0]))
solver.solve()

sph_path = "./data/tt_sph.h5"
if os.path.exists(sph_path):
    os.remove(sph_path)
inv = TraveltimeInventory(sph_path, mode="w")
inv.add(solver.traveltime, "XX/SPH/P", station_coords=solver.src_loc)  # 900 km default
inv.add(solver.traveltime, "XX/SPH/full", max_dist=None, compress=False)
f5 = inv.f5
n_lim = np.prod(f5["XX/SPH/P/npts"][:])
n_full = np.prod(f5["XX/SPH/full/npts"][:])
print(f"  nodes: full = {n_full}  limited = {n_lim}  ({n_full/n_lim:.1f}x fewer)")
fld = inv.read("XX/SPH/P")
# value at ~800 km south of station must be finite; ~1100 km must be NaN/inf
near = geo2sph(np.array([52.5 - 800 / 111.19, 32.5, 10.0]))
far = geo2sph(np.array([52.5 - 1100 / 111.19, 32.5, 10.0]))
v_near = fld.value(near, null=np.inf)
v_far = fld.value(far, null=np.inf)
print(f"  tt at 800 km: {v_near:.1f} s (finite: {np.isfinite(v_near)})   "
      f"tt at 1100 km: {v_far} (finite: {np.isfinite(v_far)})")
inv.f5.close()

print("\nall distance-limiting tests ran")
