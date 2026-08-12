#!/usr/bin/env python3
"""
Uncertainty calibration test for the EDT locator.

The question this answers is the only one that matters about an error
ellipsoid: does it contain the truth as often as it claims to? Nothing
else in the test suite checks that, and the reported width depends on a
long chain of tunable choices (alpha, edt_exponent / posterior exponent,
the proposal inflate factor, the contraction schedule, the eigenvalue
floors), any of which can silently scale sigma by a large factor without
breaking a single existing test.

Method: synthesise events with known hypocentres, locate them, characterise
the posterior, and measure the fraction of events whose true location falls
inside the nominal 68% and 95% confidence regions. Coverage is measured
three ways, because they fail differently:

  3-D   : Mahalanobis distance in the full km-frame covariance,
          compared against chi-square with 3 dof
  horiz : the two horizontal components only, chi-square with 2 dof
  depth : the vertical component only, chi-square with 1 dof

Results are reported separately for a well-surrounded network and a
one-sided one, since these are the two regimes where the locator behaves
completely differently and an average over both hides that.

Coverage well below nominal means the ellipsoid is too small (the dangerous
direction: analysts trust a location more than they should). Well above
nominal means it is uninformative. Both are reported; the assertions are
deliberately loose, because with a few tens of events the binomial noise on
a coverage estimate is several percent and this test should flag a broken
uncertainty scale, not a 5% miscalibration.

Note that coverage is measured about the REPORTED hypocentre (the mode),
not the posterior mean, because the mode is what gets published. On skewed
EDT posteriors the two differ, and the gap is reported as `mode-mean` so a
systematic offset is visible.

A NOTE ON GRID RESOLUTION
-------------------------
The defaults build a 41 x 61 x 61 grid: 1.00 km vertical and 5.93 km
horizontal node spacing over a 1.6-degree cap. The horizontal spacing is
about 13x the ~0.45 km errors being measured on a well-surrounded network,
which sounds alarming, so it was swept rather than assumed. Holding
everything else fixed (n=30, so +/-9% binomial noise):

  npts_h   spacing   68% 3-D   median semi-major   median horiz error
    61     5.93 km     83%          0.92 km             0.56 km
   101     3.56 km    100%          1.04 km             0.50 km
   161     2.22 km     93%          0.99 km             0.61 km

  npts_v   spacing   68% 3-D   median semi-major   median depth error
    21     2.00 km     90%          0.89 km             0.34 km
    41     1.00 km     97%          0.98 km             0.36 km
    81     0.50 km    100%          1.01 km             0.36 km

The reported sigma is stable to within a few percent across a 2.7x change
in horizontal resolution and a 4x change in vertical, so the conservative
ellipsoid documented below is not a discretisation artefact. Errors are
stable too, though the coarsest grids give marginally larger ones (grid
truncation adds a little location error), which flatters coverage
slightly -- worth knowing if you compare runs at different resolutions.

Note that this test cannot detect model-mismatch error at all: synthetic
arrivals are generated from the same interpolated traveltime fields the
locator uses, so discretisation is common-mode. That is the right design
for measuring uncertainty calibration, but it means the harness says
nothing about how a real velocity model's error propagates.

Use --npts-h / --npts-v / --half-deg to re-check any result you intend
to rely on. Build time scales with the node count: 61 -> 35 s, 161 -> 240 s
for 28 stations.


A NOTE ON EDT AND CONSERVATIVE ELLIPSOIDS
----------------------------------------
On a well-surrounded network with alpha > 0 this harness reports ~90-98%
coverage against a nominal 68%, i.e. sigma about 1.7x larger than the
errors. That is not a defect in the sampler and not a miscalibration that
can be tuned away. It is a property of the EDT likelihood under
HETEROSCEDASTIC errors, and it was measured rather than assumed:

  * The sampler reproduces dense-quadrature ground truth to 10-20%, so the
    reported sigma is the posterior's actual width.
  * Comparing the EDT posterior against the exact L2 posterior on the same
    grid, for the same events, gives EDT/L2 width ratios of 1.4-2.0. The
    L2 sigma matches the empirical RMS error; the EDT sigma does not.
  * With alpha = 0, so that every arrival has the same assumed sigma, EDT
    coverage returns to ~80%/90% against nominal 68%/95% -- calibrated
    within binomial noise.

The mechanism: L2 weights each arrival by 1/sigma_i^2, so four near-source
stations at sigma = 0.05 s carry ~20x the weight of eight ring stations at
sigma = 0.24 s. EDT instead sums over PAIRS, and a pair's width is
sqrt(s_a^2 + s_b^2) -- dominated by the worse of the two. Only 6 of 66
pairs join two precise stations, so most of the precision is diluted. EDT
buys outlier robustness and pays for it in efficiency whenever the assumed
errors are uneven, and NonLinLoc has the same property.

The reported ellipsoid is therefore CONSERVATIVE in that regime, which is
the safe direction. `--sigma-spread-ok` controls when over-coverage is
treated as a defect: above that spread it is reported but not failed.

Usage:
    python3 test_uncertainty_calibration.py                 # both geometries
    python3 test_uncertainty_calibration.py --n 100
    python3 test_uncertainty_calibration.py --coord-sys spherical
"""

import argparse
import math
import os
import sys
import tempfile
import time

import numpy as np

import pykonal
from pykonal.constants import EARTH_RADIUS as R_EARTH
from pykonal.inventory import TraveltimeInventory
from pykonal.locate import EQLocator


# --------------------------------------------------------------------------
# chi-square quantiles (no scipy dependency)
# --------------------------------------------------------------------------
def chi2_quantile(p, dof):
    """Quantile of the chi-square distribution. Exact for dof 1 and 2,
    Wilson-Hilferty (better than 0.5% here) for dof 3."""
    from statistics import NormalDist
    nd = NormalDist()
    if dof == 1:
        return nd.inv_cdf(0.5 * (1.0 + p)) ** 2
    if dof == 2:
        return -2.0 * math.log(1.0 - p)
    z = nd.inv_cdf(p)
    return dof * (1.0 - 2.0 / (9.0 * dof) + z * math.sqrt(2.0 / (9.0 * dof))) ** 3


# --------------------------------------------------------------------------
# model / network construction
# --------------------------------------------------------------------------
LAT0, LON0 = 19.5, -155.5

# (dlat_deg, dlon_deg) offsets. "surrounded" rings the target volume;
# "one_sided" puts every station in one quadrant, the geometry on which EDT
# is expected to produce a genuinely large, skewed ellipsoid.
NETWORKS = {
    # A realistic local network: a wide ring PLUS near-source stations.
    # The inner stations matter enormously and their absence is not a
    # subtle effect: with a ring alone at ~110 km, every takeoff angle is
    # near-horizontal, depth is essentially unresolved, and the locator
    # sits ~8 km too deep with an origin time ~1 s late. That is real
    # physics rather than a locator defect, but it means a ring-only
    # geometry tests almost nothing about depth calibration.
    "surrounded": [(0.9, 0.9), (0.9, -0.9), (-0.9, 0.9), (-0.9, -0.9),
                   (1.15, 0.0), (-1.15, 0.0), (0.0, 1.15), (0.0, -1.15),
                   (0.18, 0.12), (-0.15, 0.2), (0.05, -0.22), (-0.22, -0.06)],
    # The same wide ring with the inner stations removed: depth is
    # unresolvable and the locator sits ~8 km deep with a ~1 s late origin
    # time. Kept as an explicit geometry because it is the regime where an
    # honest (large) depth uncertainty matters most.
    "ring_only":  [(0.9, 0.9), (0.9, -0.9), (-0.9, 0.9), (-0.9, -0.9),
                   (1.15, 0.0), (-1.15, 0.0), (0.0, 1.15), (0.0, -1.15)],
    "one_sided":  [(0.75, 0.55), (0.95, 0.75), (1.15, 0.5), (0.8, 1.0),
                   (1.0, 0.25), (1.25, 0.9), (0.6, 0.8), (1.05, 1.15)],
}

DEPTH_MAX = 40.0        # km, model bottom


def velocity_profile(depth_km):
    return 4.0 + 0.1 * np.asarray(depth_km)


def build_model(coord_sys, tmpdir, npts_h=61, npts_v=41, half_deg=1.6):
    """Build the velocity grid, station set and traveltime inventory.

    Grid resolution matters more here than it looks. At the defaults the
    horizontal node spacing is ~5.9 km while the errors being measured on a
    well-surrounded network are ~0.45 km -- the nodes are an order of
    magnitude coarser than the quantity under test. Traveltimes between
    nodes come from trilinear interpolation, which makes the likelihood
    surface piecewise-smooth with kinks on cell boundaries, and first-order
    FMM has its own truncation error on top. Both are common-mode here
    (synthetic arrivals are generated from the same interpolated fields, so
    there is no model mismatch) but they are not guaranteed to cancel in
    the SHAPE of the posterior. Use --npts-h/--npts-v to check whether a
    result is resolution-dependent before trusting it.

    Returns (inv_path, stations, tt_fields, to_grid, from_grid, min_c, max_c)
    where to_grid maps (lat, lon, depth_km) -> grid coordinates.
    """
    if coord_sys == "spherical":
        th0, ph0 = math.radians(90 - LAT0), math.radians(LON0)
        dth = math.radians(half_deg)
        npts = (npts_v, npts_h, npts_h)
        node = np.array([DEPTH_MAX / (npts_v - 1),
                         2 * dth / (npts_h - 1),
                         2 * dth / (npts_h - 1)])
        mins = np.array([R_EARTH - DEPTH_MAX, th0 - dth, ph0 - dth])
        # index 0 is r, increasing upward -> depth decreases
        depths = np.linspace(DEPTH_MAX, 0.0, npts_v)
        vel = np.broadcast_to(
            velocity_profile(depths)[:, None, None], npts).copy()

        def to_grid(lat, lon, dep):
            return np.array([R_EARTH - dep,
                             math.radians(90 - lat),
                             math.radians(lon)])
    else:
        # local cartesian: x=east km, y=north km, z=depth km
        span = half_deg * 111.19
        npts = (npts_h, npts_h, npts_v)
        node = np.array([2 * span / (npts_h - 1),
                         2 * span / (npts_h - 1),
                         DEPTH_MAX / (npts_v - 1)])
        mins = np.array([-span, -span, 0.0])
        depths = np.arange(npts_v) * node[2]
        vel = np.broadcast_to(velocity_profile(depths), npts).copy()

        def to_grid(lat, lon, dep):
            return np.array([(lon - LON0) * 111.19 * math.cos(math.radians(LAT0)),
                             (lat - LAT0) * 111.19,
                             dep])

    maxs = mins + node * (np.array(npts) - 1)

    stations = {}
    for name, offsets in NETWORKS.items():
        for i, (dla, dlo) in enumerate(offsets):
            stations[(name[:2].upper(), f"{name[:1].upper()}{i:02d}")] = (
                to_grid(LAT0 + dla, LON0 + dlo, 0.0)
            )

    inv_path = os.path.join(tmpdir, f"tt_{coord_sys}.h5")
    inv = TraveltimeInventory(inv_path, mode="w")
    tt_fields = {}
    t_start = time.time()
    for (net, sta), coords in stations.items():
        solver = pykonal.solver.PointSourceSolver(coord_sys=coord_sys)
        solver.velocity.min_coords = mins
        solver.velocity.node_intervals = node
        solver.velocity.npts = npts
        solver.velocity.values = vel
        solver.src_loc = np.asarray(coords)
        solver.solve()
        key = "/".join([net, sta, "P"])
        inv.add(solver.traveltime, key)
        tt_fields[(net, sta, "P")] = solver.traveltime
    inv.f5.close()
    if coord_sys == "spherical":
        h_km = float(node[1] * R_EARTH)
        v_km = float(node[0])
    else:
        h_km = float(node[0])
        v_km = float(node[2])
    print(f"  {len(stations)} traveltime grids ({coord_sys}) in "
          f"{time.time() - t_start:.1f} s")
    print(f"  grid {'x'.join(str(int(n)) for n in npts)} "
          f"({int(np.prod(npts)):,} nodes), node spacing "
          f"{h_km:.2f} km horizontal / {v_km:.2f} km vertical")

    return inv_path, stations, tt_fields, to_grid, mins, maxs


def network_keys(tt_fields, network):
    prefix = network[:2].upper()
    return [k for k in tt_fields if k[0] == prefix]


# --------------------------------------------------------------------------
# posterior sampling with the operational contraction schedule
# --------------------------------------------------------------------------
def contracted_posterior(locator, hypo, delta, nsamples, seed, passes=6):
    """Mirror of the contraction loop used by the SeisComP extension, so the
    calibration measured here is the calibration of what actually ships."""
    d = np.abs(np.asarray(delta[:3], dtype=float)).copy()
    d0 = d.copy()
    post = None
    for _ in range(passes):
        post = locator.sample_posterior(
            hypo[:3], d, nsamples=nsamples, nscatter=0, seed=seed,
            search_delta=d0
        )
        sig = np.sqrt(np.clip(np.diag(post["covariance"]), 0, None))
        nxt = np.minimum(np.maximum(4.0 * sig, 0.35 * d), d)
        if np.all(nxt > 0.95 * d):
            break
        d = nxt
    return post


# --------------------------------------------------------------------------
# one synthetic event
# --------------------------------------------------------------------------
def run_event(locator, tt_fields, keys, true_grid, rng, args):
    t0_true = 100.0
    arrivals, pick_errors = {}, {}
    for key in keys:
        tt = tt_fields[key].value(true_grid)
        if not np.isfinite(tt) or tt > 9999:
            continue
        # Inject noise that MATCHES the error model the locator is told to
        # assume: sigma^2 = pick_sigma^2 + (alpha*tt)^2. Getting this wrong
        # invalidates the whole measurement, and it is easy to get wrong.
        # The first version of this harness injected pick noise only while
        # leaving alpha at 0.01, so at ~24 s traveltimes the locator was
        # told to expect 0.243 s of error against 0.05 s actually present.
        # It duly returned ellipsoids about five times wider than the
        # errors, and the harness reported that as a library defect. It was
        # not: with a self-consistent error model the same code came out at
        # 75% coverage against a nominal 68%.
        noise = rng.normal(0.0, args.pick_sigma)
        if args.model_error and args.alpha > 0:
            # Independent per-arrival fractional traveltime error, which is
            # exactly what the alpha term in the likelihood assumes. Real
            # velocity-model error is spatially correlated and largely
            # cancels in the differential, so this is the optimistic case;
            # see --correlated-model-error for the pessimistic one.
            noise += rng.normal(0.0, args.alpha * tt)
        arrivals[key] = t0_true + tt + noise
        pick_errors[key] = args.pick_sigma
    if len(arrivals) < 4:
        return None

    # optionally corrupt one pick, to check the ellipsoid is honest about
    # data the EDT stack is deliberately ignoring
    if args.correlated_model_error > 0:
        # A single common traveltime scale error, the correlated case. EDT
        # locates on differentials so this largely cancels and is absorbed
        # by the origin time -- which is the point: it should barely move
        # the hypocenter while the assumed sigma pays for it anyway.
        scale = 1.0 + rng.normal(0.0, args.correlated_model_error)
        for key in arrivals:
            arrivals[key] = t0_true + (arrivals[key] - t0_true) * scale

    if args.outlier_frac > 0 and rng.random() < args.outlier_frac:
        bad = list(arrivals)[rng.integers(len(arrivals))]
        arrivals[bad] += rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 5.0)

    # Spread of the ASSUMED per-arrival sigma across the network. This is
    # the quantity that controls how much wider the EDT posterior is than
    # the least-squares one (see the note in the module docstring): EDT
    # pairs every precise observation with imprecise ones, and each pair's
    # width is set by sqrt(s_a^2 + s_b^2), so a near-source station's
    # precision is diluted by whatever it is paired against.
    _sig = np.array([
        math.sqrt(args.pick_sigma ** 2 + (args.alpha * tt_fields[k].value(true_grid)) ** 2)
        for k in arrivals
    ])
    sigma_spread = float(_sig.max() / max(_sig.min(), 1e-9))

    locator.clear_arrivals()
    locator.add_arrivals(arrivals)
    locator.add_pick_errors(pick_errors)

    # start from a deliberately displaced anchor
    scale = locator._metric_scale(true_grid)
    offset_km = rng.normal(0.0, args.start_offset_km, 3)
    initial = np.append(true_grid + offset_km / scale, t0_true - 1.0)
    initial = np.clip(initial[:3], args.min_c, args.max_c)
    initial = np.append(initial, t0_true - 1.0)

    delta = np.append(args.delta_grid, 10.0)

    soln = locator.locate(initial, delta, args.alpha, "edt")
    if not np.all(np.isfinite(soln)):
        return None

    post = contracted_posterior(
        locator, soln, delta, args.nsamples, args.posterior_seed
    )

    s = locator._metric_scale(soln[:3])
    err_km = (true_grid - soln[:3]) * s          # truth relative to the mode
    mode_minus_mean_km = (soln[:3] - post["mean"]) * s
    cov = post["covariance_km"]

    # ridge-regularise before inverting: a posterior with a genuinely tiny
    # eigenvalue is not a reason to crash the whole run
    cov_r = cov + np.eye(3) * max(1e-6, 1e-9 * np.trace(cov))
    try:
        d2_3d = float(err_km @ np.linalg.solve(cov_r, err_km))
    except np.linalg.LinAlgError:
        return None

    if locator.coord_sys == "spherical":
        h_idx, v_idx = [1, 2], 0        # (theta, phi) horizontal, r vertical
    else:
        h_idx, v_idx = [0, 1], 2
    ch = cov_r[np.ix_(h_idx, h_idx)]
    eh = err_km[h_idx]
    d2_h = float(eh @ np.linalg.solve(ch, eh))
    d2_z = float(err_km[v_idx] ** 2 / cov_r[v_idx, v_idx])

    semi = post["ellipsoid_km"]["semi_axes"]
    return {
        "box_limited": bool(post.get("box_limited", False)),
        "t0_sigma": float(post.get("time_sigma", np.nan)),
        "t0_err": float(soln[3] - t0_true),
        "d2_t0": (
            float((soln[3] - t0_true) ** 2 / post["time_sigma"] ** 2)
            if np.isfinite(post.get("time_sigma", np.nan))
            and post.get("time_sigma", 0) > 0 else np.nan
        ),
        "depth_t0_corr": float(post.get("depth_time_correlation", np.nan)),
        "sigma_spread": sigma_spread,
        "at_edge": bool(np.any(post.get("at_search_edge", False))),
        "d2_3d": d2_3d,
        "d2_h": d2_h,
        "d2_z": d2_z,
        "err_h_km": float(np.hypot(*err_km[h_idx])),
        "err_z_km": float(abs(err_km[v_idx])),
        "bias_z_km": float(err_km[v_idx]),
        "dt0": float(soln[3] - t0_true),
        "semi_major": float(semi[0]),
        "semi_minor": float(semi[2]),
        "vol": float(post["ellipsoid_km"]["volume_km3"]),
        "ess": float(post["ess"]),
        "fallback": "fallback" in post["proposal"],
        "mode_mean_km": float(np.linalg.norm(mode_minus_mean_km)),
        "nobs": len(arrivals),
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def summarise(name, rows, args):
    n = len(rows)
    out = {"network": name, "n": n}
    if n == 0:
        print(f"  {name}: no successful events")
        return out

    def cov_frac(key, dof, p):
        thr = chi2_quantile(p, dof)
        vals = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
        if not vals:
            return float("nan")
        return float(np.mean([v <= thr for v in vals]))

    for label, key, dof in (("3d", "d2_3d", 3),
                            ("horiz", "d2_h", 2),
                            ("depth", "d2_z", 1),
                            ("t0", "d2_t0", 1)):
        out[f"cov68_{label}"] = cov_frac(key, dof, 0.68)
        out[f"cov95_{label}"] = cov_frac(key, dof, 0.95)

    med = lambda k: float(np.median([r[k] for r in rows]))
    out.update({
        "err_h_km": med("err_h_km"),
        "err_z_km": med("err_z_km"),
        "bias_z_km": med("bias_z_km"),
        "semi_major_km": med("semi_major"),
        "semi_minor_km": med("semi_minor"),
        "ess": med("ess"),
        "mode_mean_km": med("mode_mean_km"),
        "dt0": med("dt0"),
        "fallback_frac": float(np.mean([r["fallback"] for r in rows])),
        "sigma_spread": float(np.median([r["sigma_spread"] for r in rows])),
        "box_limited_frac": float(np.mean([r["box_limited"] for r in rows])),
        "at_edge_frac": float(np.mean([r["at_edge"] for r in rows])),
    })

    # Coverage restricted to events whose posterior is bounded by the DATA
    # rather than by the search box. A box-limited sigma is explicitly a
    # lower bound, so counting those events in the headline coverage
    # measures the box we chose, not the locator.
    free = [r for r in rows if not r["box_limited"]]
    out["n_free"] = len(free)
    if free:
        thr = chi2_quantile(0.68, 3)
        out["cov68_3d_free"] = float(np.mean([r["d2_3d"] <= thr for r in free]))
    else:
        out["cov68_3d_free"] = float("nan")

    # binomial standard error on the 68% figure, so the reader knows how
    # much of a deviation is just sampling noise
    se = math.sqrt(0.68 * 0.32 / n)

    print(f"\n  --- {name} (n={n}) ---")
    print(f"    coverage 68% [+/-{100*se:.0f}% noise]   "
          f"3d={100*out['cov68_3d']:.0f}%  "
          f"horiz={100*out['cov68_horiz']:.0f}%  "
          f"depth={100*out['cov68_depth']:.0f}%")
    print(f"    coverage 95%                    "
          f"3d={100*out['cov95_3d']:.0f}%  "
          f"horiz={100*out['cov95_horiz']:.0f}%  "
          f"depth={100*out['cov95_depth']:.0f}%")
    print(f"    median error       horiz {out['err_h_km']:.2f} km   "
          f"depth {out['err_z_km']:.2f} km (signed {out['bias_z_km']:+.2f})   "
          f"dt0 {out['dt0']:+.3f} s")
    print(f"    median 1-sigma semi-axes   major {out['semi_major_km']:.2f} km   "
          f"minor {out['semi_minor_km']:.2f} km")
    print(f"    median ess {out['ess']:.0f} of {args.nsamples}   "
          f"uniform-fallback {100*out['fallback_frac']:.0f}%   "
          f"|mode-mean| {out['mode_mean_km']:.2f} km")
    print(f"    origin time: coverage 68% {100*out['cov68_t0']:.0f}%  "
          f"median |err| {np.median([abs(r['t0_err']) for r in rows]):.3f} s  "
          f"median sigma {np.median([r['t0_sigma'] for r in rows]):.3f} s  "
          f"median depth-t0 corr "
          f"{np.median([r['depth_t0_corr'] for r in rows]):+.2f}")
    print(f"    assumed-sigma spread across network "
          f"{out['sigma_spread']:.1f}x "
          f"(EDT posterior widens with this; see module docstring)")
    print(f"    box-limited {100*out['box_limited_frac']:.0f}%   "
          f"solution-at-search-edge {100*out['at_edge_frac']:.0f}%   "
          f"68% 3-D coverage among data-limited events "
          f"{100*out['cov68_3d_free']:.0f}% (n={out['n_free']})")
    return out


def check(out, args, failures, miscal, notes):
    """Two classes of problem, reported separately.

    `failures` are signs the machinery is broken: degenerate posteriors,
    collapsed effective sample size, a proposal that never works.

    `miscal` are calibration defects: the sampler ran fine and produced a
    confident number that is the wrong size. These are looser and noisier,
    but they are the ones that mislead an analyst.
    """
    name = out["network"]
    if out["n"] < max(5, args.n // 4):
        failures.append(f"{name}: only {out['n']} events completed")
        return
    # Judge coverage on the events whose uncertainty the data actually
    # determined. Box-limited events are reported but not failed: their
    # sigma is a declared lower bound, and no change to the sampler can
    # make it correct.
    c = out["cov68_3d_free"] if out["n_free"] >= 10 else out["cov68_3d"]
    if not np.isfinite(c):
        return
    if c < args.min_coverage:
        failures.append(
            f"{name}: 68% 3-D coverage {100*c:.0f}% is far below nominal "
            f"-- ellipsoid is too small (over-confident locations)")
    if out["semi_minor_km"] <= 0.0:
        failures.append(
            f"{name}: median minor semi-axis is zero -- degenerate posterior")
    if out["fallback_frac"] > 0.5:
        failures.append(
            f"{name}: hessian proposal fell back to uniform on "
            f"{100*out['fallback_frac']:.0f}% of events")
    if out["ess"] < 30:
        failures.append(
            f"{name}: median effective sample size {out['ess']:.0f} is too "
            f"low for the covariance to mean anything")

    # --- calibration ---
    # When most events are box-limited, the reported sigma is a declared
    # LOWER BOUND: the pdf runs to the edge of the searched volume (or of
    # the velocity model) and is truncated there, so coverage below nominal
    # is arithmetically forced and says nothing about the locator. Verified
    # directly -- widening the box on these two geometries moved 68% 3-D
    # coverage from 50%/44% to 73%/70% and 95% coverage from 79%/81% to
    # 97%/100%, i.e. onto nominal. These are therefore reported as
    # unresolved rather than failed.
    if out["box_limited_frac"] > 0.5:
        notes.append(
            f"{name}: {100*out['box_limited_frac']:.0f}% of events are "
            f"box-limited and {100*out['at_edge_frac']:.0f}% sit on a search "
            f"boundary, so coverage (68% 3-D {100*out['cov68_3d']:.0f}%, "
            f"depth {100*out['cov68_depth']:.0f}%) is bounded by the search "
            f"volume, not by the data. Widen search_delta_km -- or accept "
            f"that this geometry does not determine depth")
        return

    if out["cov95_3d"] < args.min_cov95:
        miscal.append(
            f"{name}: 95% 3-D coverage is only {100*out['cov95_3d']:.0f}% "
            f"-- the posterior tails are too thin, so gross mislocations are "
            f"reported with tight ellipsoids")
    if out["cov68_depth"] < args.min_cov_depth:
        miscal.append(
            f"{name}: 68% depth coverage is only {100*out['cov68_depth']:.0f}% "
            f"(median depth error {out['err_z_km']:.1f} km, signed "
            f"{out['bias_z_km']:+.1f} km) -- depth uncertainty is understated")
    if c > args.max_coverage and out["sigma_spread"] <= args.sigma_spread_ok:
        miscal.append(
            f"{name}: 68% 3-D coverage is {100*c:.0f}% (median error "
            f"{out['err_h_km']:.2f} km horiz / {out['err_z_km']:.2f} km depth "
            f"vs median semi-major {out['semi_major_km']:.2f} km) -- the "
            f"ellipsoid is much larger than the errors warrant")
    if (out["cov68_horiz"] > args.max_cov_horiz
            and out["sigma_spread"] <= args.sigma_spread_ok):
        miscal.append(
            f"{name}: 68% horizontal coverage is {100*out['cov68_horiz']:.0f}% "
            f"-- the horizontal ellipse is larger than the errors warrant")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40,
                    help="events per network geometry (default 40)")
    ap.add_argument("--npts-h", type=int, default=61,
                    help="nodes per horizontal axis (default 61 ~ 5.9 km "
                         "spacing); raise to test resolution dependence")
    ap.add_argument("--npts-v", type=int, default=41,
                    help="nodes on the vertical axis (default 41 = 1 km)")
    ap.add_argument("--half-deg", type=float, default=1.6,
                    help="half-width of the model cap in degrees")
    ap.add_argument("--coord-sys", default="spherical",
                    choices=["spherical", "cartesian", "both"])
    ap.add_argument("--networks", default="surrounded,ring_only,one_sided",
                    help="comma-separated: surrounded, ring_only, one_sided")
    ap.add_argument("--pick-sigma", type=float, default=0.05,
                    help="pick noise, seconds (also the reported pick error)")
    ap.add_argument("--outlier-frac", type=float, default=0.0,
                    help="fraction of events given one grossly bad pick")
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="fractional traveltime error assumed AND injected")
    ap.add_argument("--no-model-error", dest="model_error",
                    action="store_false",
                    help="assume alpha but do not inject matching traveltime "
                         "error -- deliberately inconsistent, for showing "
                         "what an over-stated error model does to coverage")
    ap.add_argument("--correlated-model-error", type=float, default=0.0,
                    help="fractional traveltime error common to all arrivals")
    ap.add_argument("--nsamples", type=int, default=2048)
    ap.add_argument("--start-offset-km", type=float, default=8.0,
                    help="1-sigma displacement of the search anchor")
    ap.add_argument("--search-delta-km", type=float, nargs=3,
                    default=[30.0, 30.0, 15.0],
                    help="search half-widths (north, east, depth)")
    ap.add_argument("--posterior-seed", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20240101)
    ap.add_argument("--no-ot-wt", action="store_true",
                    help="disable EDT_OT_WT (plain EDT pair sum)")
    ap.add_argument("--edt-exponent", type=float, default=None,
                    help="override the EDT exponent (default: arrival count)")
    ap.add_argument("--min-coverage", type=float, default=0.40)
    ap.add_argument("--max-coverage", type=float, default=0.95)
    ap.add_argument("--min-cov95", type=float, default=0.85,
                    help="minimum acceptable 95%% 3-D coverage")
    ap.add_argument("--min-cov-depth", type=float, default=0.45,
                    help="minimum acceptable 68%% depth coverage")
    ap.add_argument("--sigma-spread-ok", type=float, default=2.0,
                    help="over-coverage is only treated as a defect when the "
                         "assumed sigma spread is below this; above it, EDT "
                         "is expected to be conservative (see docstring)")
    ap.add_argument("--max-cov-horiz", type=float, default=0.90,
                    help="maximum acceptable 68%% horizontal coverage")
    args = ap.parse_args()

    coord_systems = (["spherical", "cartesian"]
                     if args.coord_sys == "both" else [args.coord_sys])
    networks = [s.strip() for s in args.networks.split(",") if s.strip()]

    failures, miscal, notes = [], [], []
    t_start = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        for coord_sys in coord_systems:
            print(f"\n=== {coord_sys} "
                  f"(edt_ot_wt={'off' if args.no_ot_wt else 'on'}) ===")
            (inv_path, stations, tt_fields,
             to_grid, min_c, max_c) = build_model(
                 coord_sys, tmpdir, npts_h=args.npts_h,
                 npts_v=args.npts_v, half_deg=args.half_deg)

            # search half-widths converted to grid units
            dn, de, dz = args.search_delta_km
            if coord_sys == "spherical":
                delta_grid = np.array([dz, dn / R_EARTH, de / R_EARTH])
            else:
                delta_grid = np.array([de, dn, dz])
            args.delta_grid = delta_grid
            args.min_c, args.max_c = min_c, max_c

            for network in networks:
                keys = network_keys(tt_fields, network)
                rng = np.random.default_rng(args.seed + hash(network) % 9973)
                rows = []
                with EQLocator(inv_path, coord_sys=coord_sys) as locator:
                    locator.add_stations(stations)
                    locator.edt_ot_wt = not args.no_ot_wt
                    if args.edt_exponent is not None:
                        locator.edt_exponent = args.edt_exponent
                    for i in range(args.n):
                        lat = LAT0 + rng.uniform(-0.25, 0.25)
                        lon = LON0 + rng.uniform(-0.25, 0.25)
                        dep = rng.uniform(3.0, 20.0)
                        row = run_event(locator, tt_fields, keys,
                                        to_grid(lat, lon, dep), rng, args)
                        if row is not None:
                            rows.append(row)
                        if (i + 1) % 10 == 0:
                            print(f"    {network}: {i+1}/{args.n} "
                                  f"({time.time()-t_start:.0f} s)")
                out = summarise(f"{coord_sys}/{network}", rows, args)
                check(out, args, failures, miscal, notes)

    print(f"\ntotal {time.time() - t_start:.0f} s")
    if failures:
        print("\nBROKEN")
        for f in failures:
            print(f"  - {f}")
    if miscal:
        print("\nMISCALIBRATED")
        for f in miscal:
            print(f"  - {f}")
        print("\n  These are not sampler bugs: the posterior was sampled\n"
              "  cleanly and is simply the wrong width. Check, in order:\n"
              "  (1) whether the events are box-limited -- if so the number\n"
              "  is a declared lower bound and widening search_delta_km\n"
              "  will widen it; (2) whether the assumed error model (alpha,\n"
              "  pick errors) matches the noise actually present; (3) the\n"
              "  posterior exponent, currently the arrival count as in NLL.\n"
              "  NOT the absence of origin time from the posterior: the EDT\n"
              "  likelihood is origin-time independent by construction, so\n"
              "  the spatial pdf already marginalises over t0. Adding t0\n"
              "  would give a t0 uncertainty and a depth-t0 covariance to\n"
              "  report, but would not change spatial coverage at all.")
    if notes:
        print("\nUNRESOLVED (not a defect -- the search volume, not the "
              "locator, is setting these)")
        for f in notes:
            print(f"  - {f}")
    if failures or miscal:
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
