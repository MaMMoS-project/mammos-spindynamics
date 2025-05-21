"""Test db lookup."""

import numpy as np
import pathlib
import pytest

from mammos_spindynamics.db import get_spontaneous_magnetisation

DATA_DIR = pathlib.Path(__file__).parent.resolve() / "data"


def test_Co2Fe2H4():
    """Test material `Co2Fe2H4`.

    There is only one material with formula `Co2Fe2H4`, so this
    test should load its table without issues.
    """
    magnetisation_data = get_spontaneous_magnetisation(
        chemical_formula="Co2Fe2H4", print_info=False
    )
    assert np.allclose(
        magnetisation_data.dataframe["T[K]"], magnetisation_data.entity_map["T"].value
    )
    assert np.allclose(
        magnetisation_data.dataframe["M[A/m]"],
        magnetisation_data.entity_map["Ms"].value,
    )


def test_NdFe14B():
    """Test material `NdFe14B`.

    There is no material with such formula in the database,
    so we expect a `LookupError`.
    """
    with pytest.raises(LookupError):
        get_spontaneous_magnetisation(chemical_formula="NdFe14B")


def test_Co2Fe2H4_11():
    """Test material `Co2Fe2H4` with space group number 11.

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
