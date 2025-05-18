"""Test db lookup."""

import numpy as np
import pathlib
import pytest

from mammos_spindynamics.db import get_spontaneous_magnetisation

DATA_DIR = pathlib.Path(__file__).parent.resolve() / "data"


def test_CrNiP():
    """Test material `CrNiP`.

    There is only one material with formula `CrNiP`, so this
    test should load its table without issues.
    """
    M = get_spontaneous_magnetisation(chemical_formula="CrNiP", print_info=False)
    assert np.allclose(M(400), 186.92772987574517)
    assert np.allclose(M(450), 0.5 * (186.92772987574517 + 205.8706929415473))


def test_NdFe14B():
    """Test material `NdFe14B`.

    There is no material with such formula in the database,
    so we expect a `LookupError`.
    """
    with pytest.raises(LookupError):
        get_spontaneous_magnetisation(chemical_formula="NdFe14B")


def test_CrNiP_11():
    """Test material `CrNiP` with space group number 11.

    There is no material with such formula and space group
    in the database, so we expect a `LookupError`.
    """
    with pytest.raises(LookupError):
        get_spontaneous_magnetisation(
            chemical_formula="Co2Fe2H4", space_group_number=11
        )


def test_all():
    """Test search with no filters.

    This will select all entries in the database,
    so we expect a `LookupError`.
    """
    with pytest.raises(LookupError):
        get_spontaneous_magnetisation()


def test_uppasd_known():
    """Test query with UppASD input files.

    We expect this test to pass and retrieve a known material.
    """
    M = get_spontaneous_magnetisation(
        jfile=DATA_DIR / "known_material" / "jfile",
        momfile=DATA_DIR / "known_material" / "momfile",
        posfile=DATA_DIR / "known_material" / "posfile",
        print_info=False,
    )
    assert np.allclose(M(400), 990071.556207981)
    assert np.allclose(M(450), 0.5 * (967365.0340483808 + 957943.7467350367))


def test_uppasd_unknown():
    """Test query with UppASD input files.

    We expect this test to fail with a `LookupError`.
    """
    with pytest.raises(LookupError):
        get_spontaneous_magnetisation(
            jfile=DATA_DIR / "unknown_material" / "jfile",
            momfile=DATA_DIR / "unknown_material" / "momfile",
            posfile=DATA_DIR / "unknown_material" / "posfile",
        )


def test_uppasd_incorrect():
    """Test query with UppASD input files.

    We expect this test to fail with a `SyntaxError`.
    """
    with pytest.raises(SyntaxError):
        get_spontaneous_magnetisation(
            jfile=DATA_DIR / "wrong_data" / "jfile",
            momfile=DATA_DIR / "wrong_data" / "momfile",
            posfile=DATA_DIR / "wrong_data" / "posfile",
        )
