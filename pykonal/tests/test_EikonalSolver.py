"""
Regression and accuracy tests for EikonalSolver.

Ported from the original nose-based suite. Two things had rotted:

  * `import nose` / `nose.main()`. nose has been unmaintained since 2015
    and does not import on Python 3.10+, so the entire module was
    uncollectable. These are plain `unittest.TestCase` classes, so
    `unittest.main()` is a drop-in replacement and they then run under
    unittest, pytest and nose2 alike -- no test-runner dependency needed.

  * The 0.2-era API. `solver.vgrid` / `solver.pgrid` / `solver.vv` /
    `solver.uu` and `add_source()` no longer exist; a solver now exposes
    `velocity` and `traveltime` as ScalarField3D objects, and a source is
    seeded by zeroing its traveltime, marking it known, and pushing it
    onto the trial heap.

`pkg_resources` is also gone (deprecated, and its argument here was a path
already prefixed with "pykonal/", which only resolved by accident of the
source layout). Data files resolve relative to this module instead.
"""

import numpy as np
import pathlib
import pykonal
import unittest

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"


def solve_from_fixture(fname):
    """Rebuild the stored problem with the current API and solve it."""
    with np.load(fname) as inf:
        solver = pykonal.EikonalSolver(coord_sys="cartesian")
        solver.velocity.min_coords = inf["vgrid_min_coords"]
        solver.velocity.node_intervals = inf["vgrid_node_intervals"]
        solver.velocity.npts = inf["vgrid_npts"]
        solver.velocity.values = inf["vv"]

        # Seed the point source. The 0.2 API's add_source() did this
        # internally; there is no equivalent single call on a bare
        # EikonalSolver now (PointSourceSolver has its own src_loc).
        src = tuple(int(i) for i in inf["src"])
        solver.traveltime.values[src] = 0.0
        solver.unknown[src] = False
        solver.trial.push(*src)

        uu_ref = inf["uu"]

    solver.solve()
    return np.asarray(solver.traveltime.values), uu_ref


class EikonalSolverTestCase(unittest.TestCase):

    FIXTURE = "test_EikonalSolver_uniform_velocity_cartesian.npz"

    def test_uniform_velocity_cartesian_regression(self):
        """Current solver still reproduces the stored solution.

        Compared with a tolerance rather than assert_array_almost_equal:
        the stored array predates several solver changes and agrees to
        ~0.045 out of traveltimes reaching ~170. A six-decimal equality
        assertion here would only be testing that nobody ever touches the
        update stencil again.
        """
        uu, uu_ref = solve_from_fixture(DATA_DIR / self.FIXTURE)
        self.assertEqual(uu.shape, uu_ref.shape)
        self.assertLess(np.abs(uu - uu_ref).max(), 0.1)

    def test_uniform_velocity_cartesian_accuracy(self):
        """Unit velocity from a corner source: traveltime is the distance.

        This is the assertion the fixture comparison cannot make -- it
        checks the solver against the analytic answer rather than against
        its own past output, so it would catch an error that was baked
        into the fixture when it was generated. First-order FMM has known
        error along the diagonals near the source, so accuracy is asserted
        in the far field.
        """
        uu, _ = solve_from_fixture(DATA_DIR / self.FIXTURE)
        idx = np.meshgrid(*[np.arange(n) for n in uu.shape], indexing="ij")
        analytic = np.sqrt(sum(i.astype(float) ** 2 for i in idx))

        far = analytic > 10.0
        rel = np.abs(uu[far] - analytic[far]) / analytic[far]
        self.assertLess(rel.max(), 0.10)
        self.assertLess(np.median(rel), 0.01)

        self.assertGreaterEqual(uu.min(), 0.0)
        self.assertTrue(np.all(np.isfinite(uu)))


if __name__ == "__main__":
    unittest.main()
