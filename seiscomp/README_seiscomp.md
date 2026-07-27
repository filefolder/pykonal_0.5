# pykonal EDT locator in SeisComP via LocExt

Run the pykonal_0.5 EDT locator as a locator profile inside scolv/screloc,
side by side with NonLinLoc, without writing any C++. Uses SeisComP's
ExternalLocator (`locext` plugin): SeisComP pipes the origin + picks as XML
to a script's stdin and reads a relocated origin back from stdout.

Requires SeisComP >= 4 (LocExt), tested API against SeisComP 6/7 docs.


## 1. Python environment

The script needs BOTH the seiscomp Python bindings and your compiled
pykonal_0.5 in the same interpreter. The simplest recipe on a SeisComP
machine (bindings usually live in the system python):

    # as the sysop user
    python3 -m venv --system-site-packages ~/pykonal-venv
    source ~/pykonal-venv/bin/activate
    pip install cython numpy scipy h5py geographiclib
    cd /path/to/pykonal_0.5 && pip install .

    # verify both import cleanly:
    python -c "import seiscomp.datamodel, pykonal.locate; print('ok')"

If `import seiscomp` fails inside the venv, make sure SeisComP's python
path is visible, e.g. add to the venv activate script:
    export PYTHONPATH=$PYTHONPATH:$HOME/seiscomp/lib/python


## 2. Files

Place (and make executable where relevant):

    ~/pykonal/pykonal_locext.py       the wrapper script
    ~/pykonal/pykonal_locext.json     its configuration (edit paths!)
    ~/pykonal/traveltimes.h5          your TraveltimeInventory
    ~/pykonal/stations.csv            optional: NET,STA,lat,lon,elev_m

Important config choices in pykonal_locext.json:

  * coord_sys must match how the traveltime grids were built:
      - "spherical" for grids built via pykonal.transformations.geo2sph
        (recommended: projection-free at any aperture)
      - "cartesian" for local km grids; set ref_lat/ref_lon to the
        geographic point at grid (x=0, y=0). Geographic <-> grid mapping
        uses an azimuthal equidistant projection about that reference,
        computed with exact WGS84 geodesics (geographiclib): ranges and
        bearings from the reference are exact at any latitude, and
        point-to-point distortion stays below ~50 m over a ~450 km
        aperture. No fixed km-per-degree constants are used anywhere.
        If geographiclib is not installed, a WGS84 radii-of-curvature
        fallback is used (still latitude-dependent; sub-km at +/-2 deg).
        NOTE: build your cartesian grids (station source positions) with
        this same projection so grid and geographic frames agree — the
        script places stations via the identical transform when
        stations_csv is provided.
  * search_delta_km: half-widths of the search box around the initial
    (scautoloc / operator) origin. Keep it inside your grid volume.
  * method: "edt" (recommended) or "l1".

Test the script standalone before wiring it into SeisComP. Dump one origin
with picks from your database and pipe it through:

    scxmldump -d mysql://sysop:sysop@localhost/seiscomp \
              -O <originID> -P -f > /tmp/testorigin.xml
    cat /tmp/testorigin.xml | \
        ~/pykonal-venv/bin/python ~/pykonal/pykonal_locext.py \
        --config ~/pykonal/pykonal_locext.json > /tmp/result.xml

You should get an <origin> document on stdout and diagnostics on stderr.


## 3. SeisComP configuration

In ~/.seiscomp/global.cfg (or scolv.cfg for scolv only):

    plugins = ${plugins}, locext

    ExternalLocator.profiles = pykonalEDT:"/home/sysop/pykonal-venv/bin/python /home/sysop/pykonal/pykonal_locext.py --config /home/sysop/pykonal/pykonal_locext.json"

Notes:
  * The profile string is the full command line, so point it at the venv
    python explicitly and pass --config there. You can define several
    profiles (e.g. pykonalEDT and pykonalL1 with different config files).
  * No restart of the whole system needed; just restart scolv.


## 4. On-the-fly traveltime grids for new stations

Unlike the NonLinLoc workflow (where adding a station means manually
re-running Grid2Time), the wrapper is self-maintaining IF you configure
"velocity_models" in the JSON: before each location, any station/phase
present in the picks but missing from the traveltime inventory is
computed once with the FMM solver (using station coordinates from
stations_csv) and stored, distance-limited and compressed, into the same
inventory. All subsequent locations — from any process — find it
precomputed; the check costs ~1 ms when nothing is missing.

Concurrency is handled with a lock file (<inventory>.lock): computation
holds an exclusive lock, locating holds a shared lock, so simultaneous
scolv/screloc invocations neither collide on the HDF5 file nor solve the
same grid twice. Expect the FIRST location involving a new station to
take longer (one FMM solve per missing grid — seconds to minutes
depending on grid size); everything after is normal speed. When SEVERAL
grids are missing at once (e.g. a network expansion, or bootstrapping an
empty inventory), the solves run in parallel across CPU cores
("ensure_nproc" in the config; default uses all cores), and the solves
happen with NO lock held — concurrent locations against the existing
inventory proceed unblocked while new grids compute, with only the brief
final write serialized. Stations
absent from stations_csv, or phases without a velocity model, cannot be
computed: their arrivals are excluded with a warning in the log rather
than failing the location.

## 5. Comparing pykonal vs NonLinLoc in scolv

With both the `locnll` and `locext` plugins loaded, both locators appear
in the locator combo box (bottom left of the Location tab). The workflow
for a quick A/B on any event:

  1. Load the event, review picks as usual.
  2. Select locator "NonLinLoc" + your NLL profile, press Relocate.
     A new (uncommitted) origin appears; note RMS, gap, ellipsoid in the
     Information panel.
  3. Select locator "External" and the profile "pykonalEDT",
     press Relocate again. This relocates from the same arrival set.
  4. Toggle between the two candidate origins in the origin list of the
     Event tab (or use undo/redo buttons in the Location tab) to compare
     hypocenters, residual columns in the arrival table, RMS, azimuthal
     gap, and uncertainty.
  5. Commit whichever you prefer; both remain in the database as origins
     of the event for later systematic comparison.

Because both locators consume identical picks and report into the same
Origin fields (standardError, azimuthalGap, confidenceEllipsoid,
per-arrival timeResidual), the comparison is apples-to-apples. The
pykonal origins carry methodID "pykonal-EDT" and your configured
earthModelID, so they are easy to select later in SQL/scxmldump for bulk
statistics.

For batch comparison over many events, run screloc twice with
--locator NonLinLoc and --locator External --profile pykonalEDT
(test mode: --test -O originIDfile) and compare the resulting origins.

Fixed-depth relocation and the distance cutoff in scolv's locator
settings are honored (--fixed-depth, --max-dist; the latter requires
stations_csv so the script knows station coordinates).


## 6. What lands where in scolv

  pykonal quantity                    -> SeisComP Origin field
  ----------------------------------------------------------------
  EDT hypocenter + median t0          -> latitude/longitude/depth/time
  per-arrival residuals               -> Arrival.timeResidual (table)
  quality: rms                        -> quality.standardError
  quality: azimuthal gap              -> quality.azimuthalGap
  quality: min station distance      -> quality.minimumDistance
  posterior ellipsoid (1-sigma)       -> uncertainty.confidenceEllipsoid
  posterior sigma_z                   -> depth.uncertainty
  posterior horizontal sigma          -> uncertainty.horizontalUncertainty

Arrivals whose traveltime grid does not cover the hypocenter (e.g.
outside a distance-limited grid, once we add those) come back with
weight 0 / timeUsed false, mirroring how NLL reports rejected phases.


## 7. Troubleshooting

  * scolv popup "no origin in result document": the output must contain
    the <Origin> DIRECTLY under <seiscomp>, NOT wrapped in
    <EventParameters>. (The INPUT is wrapped in EventParameters; the
    OUTPUT is a bare Origin — the two are deliberately asymmetric per the
    LocExt spec.) Fixed in write_output; update the wrapper if you see
    this.


  * Segfault / coredump in arrival collection (faulthandler points at
    seiscomp.datamodel pickID or arrival access): fixed in read_input,
    which now extracts all origin/pick/arrival data into plain Python
    while the parent EventParameters is alive, and returns that parent as
    a keepalive. If you see this on an older copy of the wrapper, update
    pykonal_locext.py. The seiscomp bindings free child objects when the
    parent is garbage-collected; dereferencing them afterwards crashes.


  * scolv shows "locator failed": run the standalone pipe test from
    section 2; all script errors print full tracebacks to stderr, which
    scolv logs (Settings -> process manager / console output).
  * Location snaps to search-box edge: the initial origin was poor or
    search_delta_km too small; increase it (but stay inside the grid).
  * "Only N usable picks": arrivals with weight 0 in scolv are skipped
    by design; re-enable them in the arrival table if wanted.
  * One relocation much slower than usual: probably a new station
    triggering an on-the-fly FMM solve (see section 4); check stderr for
    "computed and stored". Subsequent relocations are normal speed.
  * "No traveltime grid for N arrival(s)" in the log: those
    stations/phases are not in the inventory and could not be computed
    (no velocity_models configured, station missing from stations_csv,
    or no model for that phase).
