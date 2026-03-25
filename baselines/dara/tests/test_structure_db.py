"""Tests for structure_db.py."""

import pytest

from dara.structure_db import CODDatabase, ICSDDatabase


@pytest.fixture(scope="module")
def icsd_db():
    return ICSDDatabase()


@pytest.fixture(scope="module")
def cod_db():
    return CODDatabase()


def test_icsd_database(icsd_db):
    """Test the ICSDDatabase class."""
    with pytest.raises(NotImplementedError):
        icsd_db.get_cifs_by_chemsys("Fe-O", copy_files=False)


def test_cod_database(cod_db):
    cif_paths = cod_db.get_cifs_by_chemsys("Fe-O", copy_files=False)
    assert len(cif_paths) > 0
