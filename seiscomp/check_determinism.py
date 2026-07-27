#!/usr/bin/env python3
"""
Diagnose whether pykonal's locator is deterministic on YOUR data.
Runs the SAME arrivals through locate() 5 times with a FIXED initial
origin (no feedback). If the 5 answers differ, the locator is
nondeterministic (seed not compiled in) — rebuild pykonal. If they're
identical, the drift you see in scolv is purely from scolv feeding
results back, and the wrapper's search_anchor fix handles it.

Usage:
  ~/pykonal-venv/bin/python check_determinism.py \
      /path/to/traveltimes.h5 <coord_sys> <lat> <lon> <depth_km>

  where <lat> <lon> <depth_km> is a fixed starting origin (e.g. the
  scolv/NLL origin for the problem event), and coord_sys is
  'spherical' or 'cartesian' matching your inventory.

You must edit the ARRIVALS below to match the problem event's picks,
OR point it at a saved in.xml — simplest is to just confirm determinism
with any arrivals that produce a solution.
"""
import sys, numpy as np
from pykonal.locate import EQLocator

inv_path, coord_sys = sys.argv[1], sys.argv[2]
lat, lon, depth = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])

# Does this pykonal even have the seed property?
has_seed = hasattr(EQLocator, "locate_seed")
print(f"locate_seed property present in installed pykonal: {has_seed}")
if not has_seed:
    print(">>> Seed is NOT compiled in. If the 5 solutions below differ, "
          "that's why — rebuild pykonal.")

# EDIT THESE to the problem event's arrivals: {(net,sta,phase): time_seconds}
# Times are relative; only differences matter. Replace with real picks.
ARRIVALS = {
    # ("AU","STA1","P"): 0.00,
    # ("AU","STA2","P"): 1.53,
    # ...
}
if not ARRIVALS:
    print("\n!! Edit ARRIVALS in this script with the event's picks first.")
    sys.exit(0)

from pykonal.transformations import geo2sph
if coord_sys == "spherical":
    initial_xyz = geo2sph(np.array([lat, lon, depth]))
    delta = np.array([np.radians(2), np.radians(2), 25.0, 10.0])
    initial = np.append(initial_xyz, 0.0)
else:
    initial = np.array([lon, lat, depth, 0.0])  # adjust to your projection
    delta = np.array([50., 50., 25., 10.])

print("\n5 locates, identical fixed input, no feedback:")
for i in range(5):
    with EQLocator(inv_path, coord_sys=coord_sys) as L:
        L.add_arrivals(ARRIVALS)
        soln = L.locate(initial.copy(), delta.copy(), 0.01, "edt")
    print(f"  run {i+1}: {np.round(soln,4)}")
print("\nIf these 5 differ -> nondeterministic -> rebuild pykonal.")
print("If identical -> locator is fine; scolv feedback is the cause.")
