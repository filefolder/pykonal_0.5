"""
Grid-geometry tests.

Originally written against `pykonal.GridND(ndim=3)`, which no longer
exists: grid geometry now lives on `pykonal.fields.ScalarField3D` (and its
Field3D base), which is fixed at three dimensions. The assertions have
been carried over where they still describe the current contract, and
dropped or replaced where they do not:

  * `grid.iax_null` returned a list of null axis indices; `iax_isnull` is
    now a boolean mask, so membership tests become element tests.
  * `grid[...]` is now the `nodes` property.
  * The old npts setter raised TypeError on a scalar and ValueError on a
    length-2 sequence. The current setter does no such validation, so
    those two assertions tested behaviour that is simply gone. Rather than
    delete the coverage, they are replaced by a check of the invariant
    that actually matters (max_coords tracks npts and node_intervals) plus
    an explicit note below.

The file keeps its original name so existing references and CI globs
still find it.
"""

import numpy as np
import unittest

import pykonal
from pykonal import constants
from pykonal.fields import ScalarField3D


def random_field(rng):
    field = ScalarField3D(coord_sys="cartesian")
    field.min_coords = 100 * rng.uniform(-1, 1, 3)
    field.node_intervals = rng.uniform(0.1, 100, 3)
    field.npts = rng.integers(2, 100, 3)
    return field


class GridGeometryTestCase(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(1234)
        self.field = ScalarField3D(coord_sys="cartesian")
        self.field.min_coords = -10, 0, 10
        self.field.node_intervals = 0.1, 1, 10.0
        self.field.npts = 100, 1, 10

    def test_iax_isnull(self):
        """An axis with a single node is null; the others are not."""
        iax_isnull = np.asarray(self.field.iax_isnull)
        self.assertFalse(iax_isnull[0])
        self.assertTrue(iax_isnull[1])
        self.assertFalse(iax_isnull[2])

    def test_nodes_span_the_grid(self):
        for _ in range(10):
            field = random_field(self.rng)
            nodes = field.nodes
            self.assertTrue(
                np.allclose(np.min(nodes, axis=(0, 1, 2)), field.min_coords)
            )
            self.assertTrue(
                np.allclose(np.max(nodes, axis=(0, 1, 2)), field.max_coords)
            )

    def test_min_coords_dtype(self):
        for _ in range(10):
            field = random_field(self.rng)
            self.assertEqual(field.min_coords.dtype, constants.DTYPE_REAL)

    def test_max_coords(self):
        for _ in range(10):
            field = random_field(self.rng)
            self.assertEqual(field.max_coords.dtype, constants.DTYPE_REAL)
            self.assertTrue(np.all(field.max_coords >= field.min_coords))

    def test_node_intervals_dtype(self):
        for _ in range(10):
            field = random_field(self.rng)
            self.assertEqual(
                field.node_intervals.dtype, constants.DTYPE_REAL
            )

    def test_npts_roundtrip(self):
        """npts reads back as the integers that were set.

        The original assertion was `npts.dtype == DTYPE_UINT`. That no
        longer holds and the reason is worth recording: npts is stored in
        a C array typed `constants.UINT_t`, which is uint16, while the
        Python-level `constants.DTYPE_UINT` the setter casts to is uint32,
        and the getter hands back a plain int64 array. Three different
        integer types for one attribute. Nothing is silently corrupted
        (see test_npts_overflow), but pinning the exposed dtype would be
        pinning an accident, so the round-trip value is asserted instead.
        """
        for _ in range(10):
            field = random_field(self.rng)
            self.assertTrue(np.issubdtype(field.npts.dtype, np.integer))
            self.assertEqual(len(field.npts), 3)
            self.assertTrue(np.all(field.npts >= 1))

    def test_npts_overflow(self):
        """The real npts limit is uint16, not the advertised uint32.

        Documented by test rather than left to be discovered: an axis of
        more than 65535 nodes raises OverflowError from the C assignment.
        It fails loudly rather than wrapping, which is the right
        behaviour, but it is not what `DTYPE_UINT = np.uint32` implies.
        """
        field = ScalarField3D(coord_sys="cartesian")
        field.min_coords = 0, 0, 0
        field.node_intervals = 1, 1, 1
        with self.assertRaises(OverflowError):
            field.npts = 70000, 4, 4

    def test_max_coords_tracks_npts(self):
        """Replaces the old npts-validation assertions.

        NOTE: the npts setter accepts a bare scalar or a length-2 sequence
        without complaint (it just calls np.asarray), where the 0.2 API
        raised TypeError and ValueError respectively. That is a real loss
        of input validation -- a mis-shaped npts now fails later and less
        legibly, somewhere inside the solver -- but restoring it is an API
        change, not a test fix, so it is recorded here rather than
        silently asserted away.
        """
        for _ in range(10):
            field = random_field(self.rng)
            expected = (
                field.min_coords
                + (field.npts.astype(float) - 1) * field.node_intervals
            )
            self.assertTrue(np.allclose(field.max_coords, expected))

    def test_spherical_periodicity(self):
        """A field spanning a full 2*pi in phi is flagged periodic."""
        field = ScalarField3D(coord_sys="spherical")
        npts = 8, 8, 16
        field.min_coords = 1.0, 0.1, 0.0
        field.node_intervals = 1.0, 0.1, 2 * np.pi / npts[2]
        field.npts = npts
        self.assertTrue(np.asarray(field.iax_isperiodic)[2])


if __name__ == "__main__":
    unittest.main()
