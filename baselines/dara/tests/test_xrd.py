import tempfile
import unittest
from pathlib import Path

import numpy as np

from dara.xrd import RASXFile, rasx2xy


class TestRASXFile(unittest.TestCase):
    """Test RASX file loading and conversion."""

    def setUp(self):
        """Set up the test. This test file is taken from https://github.com/thomasgredig/rigakuXRD/blob/master/inst/extdata/RM20220915Si01.rasx"""
        self.rasx_path = Path(__file__).parent / "test_data" / "RM20220915Si01.rasx"

    def test_rasx_file_load(self):
        """Test loading a RASX file with specific values from RM20220915Si01.rasx."""
        rasx_file = RASXFile.from_file(self.rasx_path)

        # Check that angles and intensities are loaded
        self.assertIsNotNone(rasx_file.angles)
        self.assertIsNotNone(rasx_file.intensities)

        # Check that they are numpy arrays
        self.assertIsInstance(rasx_file.angles, np.ndarray)
        self.assertIsInstance(rasx_file.intensities, np.ndarray)

        # Check specific values from the file
        self.assertEqual(len(rasx_file.angles), 2151)
        self.assertEqual(len(rasx_file.intensities), 2151)

        # Check first angle and intensity values
        np.testing.assert_almost_equal(rasx_file.angles[0], 8.5, decimal=2)
        np.testing.assert_almost_equal(rasx_file.intensities[0], 591.0, decimal=0)

        # Check last angle and intensity values
        np.testing.assert_almost_equal(rasx_file.angles[-1], 30.0, decimal=2)
        np.testing.assert_almost_equal(rasx_file.intensities[-1], 524.0, decimal=0)

        # Check max intensity
        np.testing.assert_almost_equal(np.max(rasx_file.intensities), 639.0, decimal=0)

        # Check intermediate values
        np.testing.assert_almost_equal(rasx_file.angles[100], 9.5, decimal=2)
        np.testing.assert_almost_equal(rasx_file.intensities[100], 565.0, decimal=0)
        np.testing.assert_almost_equal(rasx_file.angles[500], 13.5, decimal=2)
        np.testing.assert_almost_equal(rasx_file.intensities[500], 526.0, decimal=0)
        np.testing.assert_almost_equal(rasx_file.angles[1000], 18.5, decimal=2)
        np.testing.assert_almost_equal(rasx_file.intensities[1000], 560.0, decimal=0)

        # Check that intensities are non-negative
        self.assertGreaterEqual(np.min(rasx_file.intensities), 0)

    def test_rasx_file_properties(self):
        """Test RASXFile properties."""
        rasx_file = RASXFile.from_file(self.rasx_path)

        # Check that binary_data is stored
        self.assertIsNotNone(rasx_file.binary_data)
        self.assertIsInstance(rasx_file.binary_data, bytes)
        self.assertGreater(len(rasx_file.binary_data), 0)

    def test_rasx_to_xy_conversion(self):
        """Test converting RASX to XY format with specific values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            xy_path = rasx2xy(self.rasx_path, tmpdir)

            # Check that XY file was created
            self.assertTrue(xy_path.exists())
            self.assertTrue(xy_path.is_file())
            self.assertEqual(xy_path.suffix, ".xy")

            # Check that the XY file contains valid data with correct values
            xy_data = np.loadtxt(xy_path, unpack=False)
            self.assertEqual(xy_data.shape[0], 2151)  # Should have 2151 data points
            self.assertEqual(xy_data.shape[1], 2)  # angle, intensity columns

            # Check first and last values match
            np.testing.assert_almost_equal(xy_data[0, 0], 8.5, decimal=2)  # First angle
            np.testing.assert_almost_equal(xy_data[0, 1], 591.0, decimal=0)  # First intensity
            np.testing.assert_almost_equal(xy_data[-1, 0], 30.0, decimal=2)  # Last angle
            np.testing.assert_almost_equal(xy_data[-1, 1], 524.0, decimal=0)  # Last intensity

    def test_rasx_file_round_trip(self):
        """Test that RASX file can be loaded and saved back."""
        rasx_file = RASXFile.from_file(self.rasx_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_path = tmpdir / "test_output.rasx"

            # Save the file
            rasx_file.to_rasx_file(output_path)

            # Check that file was created
            self.assertTrue(output_path.exists())

            # Load it back and compare
            loaded_file = RASXFile.from_file(output_path)

            # Check that angles and intensities match
            np.testing.assert_array_almost_equal(
                rasx_file.angles, loaded_file.angles, decimal=6
            )
            np.testing.assert_array_almost_equal(
                rasx_file.intensities, loaded_file.intensities, decimal=6
            )

    def test_rasx_file_plot(self):
        """Test that RASX file can be plotted."""
        rasx_file = RASXFile.from_file(self.rasx_path)

        # Test plotting (should not raise an exception)
        ax = rasx_file.plot()
        self.assertIsNotNone(ax)

        # Test with different styles
        ax = rasx_file.plot(style="points")
        self.assertIsNotNone(ax)

        ax = rasx_file.plot(style="line")
        self.assertIsNotNone(ax)

