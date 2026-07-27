# pykonal validation suite

`pykonal_validation.py` is a single, **no-input** script that validates and
benchmarks the beta `pykonal` package end to end, entirely in the native
**spherical** `(r, θ, φ)` frame, using a 1-D **AK135** `v(r)` profile.

```bash
python pykonal_validation.py
```

No arguments, no stdin, no data files. It assumes `pykonal` is already built
and importable. Artifacts (`report.txt`, `results.json`, `console.log`) are
written to `./pykonal_validation_output/`.

## Choosing regimes

By default both regimes run. To run only one:

```bash
python pykonal_validation.py --regimes regional
python pykonal_validation.py --regimes global
PYKONAL_TEST_REGIMES=global python pykonal_validation.py   # env-var form
```

A `--regimes` flag overrides the `PYKONAL_TEST_REGIMES` env var, which
overrides the `both` default. Regime-independent tests always run; the
locator/parameter/quality tests run against a "primary" regime (regional if
selected, else global); the teleseismic test and per-regime sweeps are gated.

- **regional** — ~400 km cap, AK135 crust + uppermost mantle (0–120 km),
  surface stations, shallow events, P and S.
- **global** — large-aperture (~75°) cap for teleseismic distances, AK135 to
  ~1500 km depth, deep events, P only.

## What it covers

0. **import / environment**.
1. **solver correctness** — homogeneous-sphere traveltime judged by
   **convergence** on a fixed shell (not a fixed error threshold);
   `PointSourceSolver` on AK135; causal stencil across aspect ratios.
2. **fields** — HDF round-trip, interpolation, out-of-grid null.
3. **inventory** — build, read-back fidelity, `max_dist` masking.
4. **EQLocator EDT vs L1** — recovery under pick noise, clean and with
   outliers; global teleseismic recovery + posterior.
4b. **performance matrix** — the main "how does the model perform" section.
   A set of named scenarios spanning station count (4→16), phase set (P vs
   P+S), pick dropout, added noise and outliers, and azimuthal coverage
   (surrounded / one-sided / 120° gap). Each scenario locates a batch of
   random events and reports median & p90 epicentral / depth / total error,
   median azimuthal gap, median `nobs`, and (where sampled) posterior ESS.
5. **parameter sweeps** — `alpha` swept **0 → 0.09 in 0.005 steps** (with the
   best-scoring alpha reported), `edt_reg` × geometry, `edt_exponent`,
   posterior proposal (Hessian vs uniform, cross-checked against a
   scatter-derived physical ellipsoid), `locate_seed` determinism, and the
   `alpha`-not-reset regression.
6. **vertical compression (precision vs model size)** — the grid study,
   reframed. Horizontal node spacing is held fixed; the **vertical (radial)
   spacing is varied** to give dx:dz ratios from 1:1 (fine vertical, larger
   model) to 10:1 (compressed vertical, smaller model). For each ratio it
   reports model size (nodes per grid and MB) alongside median/p90 **depth**
   and total error, and recommends the most-compressed grid whose depth
   precision is still within ~15% of the finest — the optimal size/precision
   knee.
7. **quality metrics** — `nobs`, `azimuthal_gap`, `min_station_dist`, `rms`.
8. **robustness** — <2 arrivals, out-of-grid arrival dropped, inventory
   double-close guard, all-outlier input.

Every test is isolated: a failure is captured with a trimmed traceback and the
run always reaches the final report.

## Runtime / scaling

`PYKONAL_TEST_LEVEL` = `quick` | `standard` (default) | `full` scales grid
sizes, station/event counts, and the number of events per performance
scenario. All sizes live in the `CONFIG` / `Regime` block at the top of the
file.

## Ellipsoids are physical km

The suite reports posterior error ellipsoids in **km** (semi-axes and volume),
computed by transforming the posterior cloud into a local Cartesian/ENU frame.
It prefers the library's `ellipsoid_km` (present after the `sample_posterior`
fix) and otherwise derives the same quantity from the scatter cloud, so results
are correct whether or not the library has been rebuilt. The raw
`ellipsoid`/`covariance` keys remain in native coordinates (km radial, radians
angular for spherical) and should not be read as physical lengths.

## Caveats

- **Inverse crime, by design.** Synthetic arrivals come from the same grids the
  locator reads, plus pick noise/outliers — this isolates the locator. Add an
  independent forward solve in `synth_arrivals` for model-error studies.
- Locator traveltime tables use the plain `EikonalSolver` (node-snapped source)
  for speed and exact resolution control; `PointSourceSolver` is exercised
  separately in section 1.
- Absolute magnitudes depend on resolution and geometry; read the performance
  matrix and the compression study **relatively**.
- The homogeneous-sphere check judges correctness by convergence, since a
  first-order fast-marching scheme has expected O(10%) near-source error.
