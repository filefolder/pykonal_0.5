import numpy as np, os, warnings, time, subprocess, sys
import pykonal
from pykonal.inventory import TraveltimeInventory, ensure_traveltimes
from pykonal.locate import EQLocator
from pykonal import fields

vm = fields.ScalarField3D(coord_sys="cartesian")
vm.min_coords = (0, 0, 0); vm.node_intervals = (2, 2, 2); vm.npts = (101, 101, 26)
vm.values = np.full((101, 101, 26), 5.0)
vm.to_hdf("./data/vp.h5")

stations = {("XX", "A"): (40., 40., 0.), ("XX", "B"): (160., 40., 0.),
            ("XX", "C"): (100., 160., 0.)}

p = "./data/ensure_test.h5"
for f in (p, p + ".lock"):
    if os.path.exists(f):
        os.remove(f)
inv = TraveltimeInventory(p, mode="w")
for sta in ("A", "B"):
    s = pykonal.solver.PointSourceSolver(coord_sys="cartesian")
    s.velocity.min_coords = (0, 0, 0); s.velocity.node_intervals = (2, 2, 2)
    s.velocity.npts = (101, 101, 26); s.velocity.values = vm.values
    s.src_loc = np.array(stations[("XX", sta)]); s.solve()
    inv.add(s.traveltime, f"XX/{sta}/P", station_coords=s.src_loc, max_dist=300)
inv.f5.close()

true = np.array([90., 90., 10.]); t0 = 50.0

# 1. BEFORE ensure: arrival from unknown station C -> warn + drop, not crash
with TraveltimeInventory(p) as i:
    arr = {k: t0 + i.read("/".join(k)).value(true)
           for k in [("XX", "A", "P"), ("XX", "B", "P")]}
arr[("XX", "C", "P")] = t0 + 25.0
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    with EQLocator(p, coord_sys="cartesian") as loc:
        loc.add_arrivals(arr)
        soln = loc.locate(np.array([100., 100., 15., 49.]),
                          np.array([60., 60., 14., 10.]), 0.01, "edt")
        used = loc.residuals(soln)
assert any("No traveltime grid" in str(x.message) for x in w)
assert len(used) == 2 and ("XX", "C", "P") not in used
print(f"1. missing-grid arrival: warned + dropped, located with {len(used)}/3  ok")

# 2. ensure() computes only the missing key
requests = {("XX", s, "P"): np.array(c) for (n, s), c in stations.items()}
tic = time.time()
rep = ensure_traveltimes(p, requests, {"P": "/tmp/vp.h5"}, max_dist=300)
dt = time.time() - tic
assert sorted(rep["present"]) == [("XX", "A", "P"), ("XX", "B", "P")]
assert rep["computed"] == [("XX", "C", "P")] and not rep["skipped"]
print(f"2. ensure(): computed C only ({dt:.1f} s), A/B untouched  ok")

# 3. second run: everything present, near-instant
tic = time.time()
rep = ensure_traveltimes(p, requests, {"P": "./data/vp.h5"}, max_dist=300)
dt = time.time() - tic
assert len(rep["present"]) == 3 and not rep["computed"]
print(f"3. second run: all present, ensure() took {dt*1000:.0f} ms  ok")

# 4. the previously failing location now uses all 3 arrivals,
#    with the true arrival time for C from its freshly stored grid
with TraveltimeInventory(p) as i:
    arr[("XX", "C", "P")] = t0 + i.read("XX/C/P").value(true)
with EQLocator(p, coord_sys="cartesian") as loc:
    loc.add_arrivals(arr)
    soln = loc.locate(np.array([100., 100., 15., 49.]),
                      np.array([60., 60., 14., 10.]), 0.01, "edt")
    used = loc.residuals(soln)
err = np.linalg.norm(soln[:3] - true)
assert len(used) == 3
print(f"4. relocate after ensure: {len(used)}/3 picks, loc err {err:.2f} km  ok")

# 5. phase with no velocity model -> skipped with warning, others proceed
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    rep = ensure_traveltimes(p, {("XX", "A", "S"): np.array(stations[("XX", "A")])},
                             {"P": "/tmp/vp.h5"}, max_dist=300)
assert rep["skipped"] == [("XX", "A", "S")]
print("5. missing S velocity model: key skipped with warning  ok")

# 6. concurrency: two processes ensure the same missing key simultaneously;
#    the flock must serialize them and exactly one computes
with TraveltimeInventory(p, mode="a") as inv:
    if inv.has("XX/D/P"):
        del inv.f5["XX/D/P"]
worker = '''
import numpy as np
from pykonal.inventory import ensure_traveltimes
rep = ensure_traveltimes("./data/ensure_test.h5",
                         {("XX","D","P"): np.array([60.,120.,0.])},
                         {"P": "/tmp/vp.h5"}, max_dist=300)
print("computed" if rep["computed"] else "present")
'''
procs = [subprocess.Popen([sys.executable, "-c", worker],
                          stdout=subprocess.PIPE, text=True) for _ in range(2)]
outs = [pr.communicate()[0].strip() for pr in procs]
assert all(pr.returncode == 0 for pr in procs), outs
assert sorted(outs) == ["computed", "present"], outs
print(f"6. concurrent ensure of same key: outcomes {outs} — computed exactly once  ok")

print("\nall ensure() tests pass")
