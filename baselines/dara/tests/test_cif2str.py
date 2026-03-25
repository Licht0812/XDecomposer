import tempfile
import unittest
import warnings
from pathlib import Path

from dara.cif2str import cif2str
from dara.utils import read_phase_name_from_str


class TestCif2Str(unittest.TestCase):
    """Test the cif2str function."""

    def setUp(self):
        """Set up the test."""
        self.cif_paths = list((Path(__file__).parent / "test_data").glob("*.cif"))

    def test_cif2str(self):
        """Test the cif2str function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for cif_path in self.cif_paths:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    str_path = cif2str(cif_path, "", tmpdir)
                    self.assertTrue(len(w) == 0)

                self.assertTrue(str_path.exists())
                self.assertTrue(str_path.is_file())
                self.assertTrue(str_path.suffix == ".str")
                self.assertTrue(str_path.stem == cif_path.stem)

                ref_str_path = cif_path.parent / (cif_path.stem + ".str")

                ref_phase_name = read_phase_name_from_str(ref_str_path)
                phase_name = read_phase_name_from_str(str_path)
                self.assertTrue(ref_phase_name == phase_name)
