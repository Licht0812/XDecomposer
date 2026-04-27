import tempfile
import unittest
from pathlib import Path

from dara.refine import do_refinement


class TestRefinement(unittest.TestCase):
    def setUp(self):
        """Set up the test."""
        self.cif_paths = list((Path(__file__).parent / "test_data").glob("*.cif"))
        self.pattern_path = Path(__file__).parent / "test_data" / "BiFeO3.xy"

    def test_refinement(self):
        """Test the refinement function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            cif_paths = self.cif_paths
            pattern_path = self.pattern_path

            result = do_refinement(
                pattern_path,
                cif_paths,
                instrument_profile="Aeris-fds-Pixcel1d-Medipix3",
                working_dir=tmpdir,
            )
            self.assertLess(result.lst_data.rwp, 8)
