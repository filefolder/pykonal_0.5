"""
End-to-end test of NLL-style features ported into pykonal_0.5:
EDT location, station corrections, posterior scatter/ellipsoid, quality metrics.
"""
import numpy as np
import pykonal
from pykonal.inventory import TraveltimeInventory
from pykonal.locate import EQLocator

rng = np.random.default_rng(42)

# ------------------------------------------------------------------ model
# 50 x 50 x 20 km cartesian volume, 0.5 km nodes, v = 4 + 0.1*z km/s
min_coords = (0.0, 0.0, 0.0)
node_intervals = (0.5, 0.5, 0.5)
npts = (101, 101, 41)
zz = np.arange(npts[2]) * node_intervals[2]
velocity = 4.0 + 0.1 * zz
velocity = np.broadcast_to(velocity, npts).copy()

# ------------------------------------------------------------------ stations
stations = {
    ("XX", "ST01"): (5.0, 5.0, 0.0),
    ("XX", "ST02"): (45.0, 5.0, 0.0),
    ("XX", "ST03"): (5.0, 45.0, 0.0),
    ("XX", "ST04"): (45.0, 45.0, 0.0),
    ("XX", "ST05"): (25.0, 10.0, 0.0),
    ("XX", "ST06"): (25.0, 40.0, 0.0),
    ("XX", "ST07"): (10.0, 25.0, 0.0),
    ("XX", "ST08"): (40.0, 25.0, 0.0),
}

# ------------------------------------------------------------------ traveltimes
inv_path = "./data/tt_inventory.h5"
inv = TraveltimeInventory(inv_path, mode="w")
tt_fields = {}
for (net, sta), coords in stations.items():
    solver = pykonal.solver.PointSourceSolver(coord_sys="cartesian")
    solver.velocity.min_coords = min_coords
    solver.velocity.node_intervals = node_intervals
    solver.velocity.npts = npts
    solver.velocity.values = velocity
    solver.src_loc = np.array(coords)
    solver.solve()
    inv.add(solver.traveltime, "/".join([net, sta, "P"]))
    tt_fields[(net, sta, "P")] = solver.traveltime
inv.f5.close()
print("traveltime inventory built")

# ------------------------------------------------------------------ synthetic event
true_hypo = np.array([22.0, 28.0, 9.0])
true_t0 = 100.0
pick_sigma = 0.05

arrivals, pick_errors = {}, {}
for key, field in tt_fields.items():
    tt = field.value(true_hypo)
    arrivals[key] = true_t0 + tt + rng.normal(0, pick_sigma)
    pick_errors[key] = pick_sigma

# inject one gross outlier (+2.5 s on ST04) — the EDT robustness test
arrivals[("XX", "ST04", "P")] += 2.5

initial = np.array([25.0, 25.0, 10.0, 99.0])
delta = np.array([15.0, 15.0, 9.0, 5.0])

def report(tag, soln):
    derr = np.linalg.norm(soln[:3] - true_hypo)
    terr = soln[3] - true_t0
    print(f"  {tag:28s} loc err = {derr:6.3f} km   t0 err = {terr:+.3f} s   "
          f"xyz = {np.round(soln[:3], 2)}")

# ------------------------------------------------------------------ locate
with EQLocator(inv_path, coord_sys="cartesian") as loc:
    loc.add_arrivals(arrivals)
    loc.add_pick_errors(pick_errors)
    loc.add_stations(stations)

    print("\nWith outlier on ST04 (+2.5 s):")
    soln_l1 = loc.locate(initial.copy(), delta.copy(), 0.01, "l1")
    report("L1 (existing)", soln_l1)
    soln_edt = loc.locate(initial.copy(), delta.copy(), 0.01, "edt")
    report("EDT (ported)", soln_edt)

    # ---------------------------------------------------------- residuals -> ID outlier
    res = loc.residuals(soln_edt)
    worst = max(res, key=lambda k: abs(res[k]))
    print(f"\n  residuals identify outlier: {worst} -> {res[worst]:+.3f} s "
          f"(others |r| < {max(abs(v) for k, v in res.items() if k != worst):.3f} s)")

    # ---------------------------------------------------------- quality metrics
    q = loc.quality(soln_edt)
    print(f"  quality: nobs={q['nobs']}  rms={q['rms']:.3f}s  "
          f"gap={q['azimuthal_gap']:.1f}deg  dmin={q['min_station_dist']:.1f}km")

    # ---------------------------------------------------------- posterior
    post = loc.sample_posterior(soln_edt[:3], [3.0, 3.0, 3.0], nsamples=4096, seed=1)
    ax = post["ellipsoid"]["semi_axes"]
    print(f"  posterior: mean={np.round(post['mean'],2)}  "
          f"1-sigma semi-axes = {np.round(ax, 2)} km  ESS={post['ess']:.0f}")
    print(f"  scatter cloud shape: {post['scatter'].shape}")
    inside = np.linalg.norm(post['mean'] - true_hypo) < 3 * ax[0] + 0.5
    print(f"  true hypocenter within ~3-sigma of posterior mean: {inside}")

    # ---------------------------------------------------------- locate_detailed
    loc.clear_arrivals(); loc.add_arrivals(arrivals)
    detail = loc.locate_detailed(initial.copy(), delta.copy(), alpha=0.01,
                                 method="edt", nsamples=2048, seed=2)
    print("\nlocate_detailed keys:", sorted(detail.keys()),
          "| posterior keys:", sorted(detail["posterior"].keys()))

print("\nall tests ran")
