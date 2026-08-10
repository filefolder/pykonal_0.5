"""
Trilinear interpolation tests.

Originally written against `pykonal.Grid3D` + `pykonal.LinearInterpolator3D`,
neither of which exists any more: interpolation is now a method on the
field itself, `ScalarField3D.value(point, null=...)`.

The out-of-bounds contract changed with it, and the change is worth being
explicit about. The 0.2 API raised `pykonal.OutOfBoundsError`; the current
one returns a caller-supplied `null` sentinel (default NaN) and raises
nothing. That is why `locate.pyx` calls `value(..., null=INFINITY)` and
then screens for non-finite results -- an out-of-range arrival is data to
be excluded, not an exception. Tests updated accordingly.

Dropped: `test_interpolate_error_float`, which filled a grid with
`pykonal.ERROR_REAL` and checked it propagated. There is no ERROR_REAL in
the current constants module and no sentinel-propagation contract to test.

The file keeps its original name so existing references still find it.
"""

import numpy as np
import unittest

from pykonal import constants
from pykonal.fields import ScalarField3D


def random_field(rng):
    field = ScalarField3D(coord_sys="cartesian")
    field.min_coords = 100 * rng.uniform(-1, 1, 3)
    field.node_intervals = rng.uniform(0.1, 100, 3)
    field.npts = rng.integers(2, 100, 3)
    values = rng.integers(1, 10000) * rng.random(tuple(field.npts))
    field.values = values.astype(constants.DTYPE_REAL)
    return field


class LinearInterpolator3DTestCase(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(4321)

    def test_interpolate_2D(self):
        """A degenerate (single-node) axis still interpolates."""
        field = ScalarField3D(coord_sys="cartesian")
        field.min_coords = 0, 0, 0
        field.node_intervals = 1, 1, 1
        field.npts = 10, 10, 1
        field.values = self.rng.random((10, 10, 1)).astype(
            constants.DTYPE_REAL
        )
        # value() takes a typed memoryview, not a Python list
        value = field.value(np.array([5.0, 5.0, 0.0], dtype=constants.DTYPE_REAL))
        self.assertTrue(np.isfinite(value))

    def test_interpolate_within_bounds(self):
        """Interior values are bracketed by the extremes of the field."""
        for _ in range(10):
            field = random_field(self.rng)
            vmin, vmax = field.values.min(), field.values.max()
            delta = field.max_coords - field.min_coords
            for _ in range(10):
                xyz = (
                    field.min_coords + self.rng.random(3) * delta
                ).astype(constants.DTYPE_REAL)
                value = field.value(xyz)
                self.assertGreaterEqual(value, vmin)
                self.assertLessEqual(value, vmax)

    def test_interpolate_is_exact_for_linear_fields(self):
        """Trilinear interpolation reproduces a linear function exactly.

        The strongest available check on the interpolation weights: any
        error in the stencil shows up immediately, whereas a bounds check
        would pass with the weights permuted.
        """
        field = ScalarField3D(coord_sys="cartesian")
        field.min_coords = -3.0, 2.0, 0.5
        field.node_intervals = 0.7, 1.3, 0.9
        field.npts = 12, 9, 15
        nodes = field.nodes
        coeff = np.array([2.0, -3.0, 0.5])
        field.values = (
            5.0 + nodes @ coeff
        ).astype(constants.DTYPE_REAL)

        delta = field.max_coords - field.min_coords
        for _ in range(50):
            xyz = (
                field.min_coords + self.rng.random(3) * delta
            ).astype(constants.DTYPE_REAL)
            expected = 5.0 + xyz @ coeff
            self.assertAlmostEqual(
                float(field.value(xyz)), float(expected), places=4
            )

    def test_out_of_bounds_returns_null(self):
        """Out of range returns the sentinel rather than raising.

        Replaces the old OutOfBoundsError assertions. The default is NaN;
        `locate.pyx` passes INFINITY so it can screen arrivals whose
        traveltime grid does not cover a trial hypocentre.
        """
        for _ in range(10):
            field = random_field(self.rng)
            delta = field.max_coords - field.min_coords
            for point in (
                field.max_coords + 2 * delta,
                field.min_coords - 2 * delta,
            ):
                point = point.astype(constants.DTYPE_REAL)
                self.assertTrue(np.isnan(field.value(point)))
                self.assertTrue(
                    np.isinf(field.value(point, null=np.inf))
                )

    def test_interpolate_edge_case_lower(self):
        for _ in range(10):
            field = random_field(self.rng)
            value = field.value(field.min_coords)
            self.assertAlmostEqual(
                float(value), float(field.values[0, 0, 0]), places=4
            )

    def test_interpolate_edge_case_upper(self):
        for _ in range(10):
            field = random_field(self.rng)
            value = field.value(field.max_coords)
            self.assertAlmostEqual(
                float(value), float(field.values[-1, -1, -1]), places=4
            )

    def test_periodic(self):
        """Interpolation wraps across the phi seam of a spherical field."""
        npts = (2, 21, 40)
        field = ScalarField3D(coord_sys="spherical")
        field.min_coords = 1, 0, 0
        field.node_intervals = 1, np.pi / 20, 2 * np.pi / npts[2]
        field.npts = npts
        field.values = np.cos(field.nodes[..., 2]).astype(
            constants.DTYPE_REAL
        )
        self.assertTrue(np.asarray(field.iax_isperiodic)[2])
        # just inside the seam: cos(phi) near phi = 2*pi is near +1
        value = field.value(np.array(
            [1.5, np.pi / 2, 39.5 * 2 * np.pi / npts[2]],
            dtype=constants.DTYPE_REAL
        ))
        self.assertTrue(np.isfinite(value))
        self.assertAlmostEqual(float(value), 1.0, places=1)


if __name__ == "__main__":
    unittest.main()
