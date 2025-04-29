"""Test db lookup."""

import pathlib
import pytest

from mammos_spindynamics.db import get_M

DATA = pathlib.Path(__file__).parent.resolve() / "data"


def test_CrNiP():
    """Test material `CrNiP`.

    There is only one material with formula `CrNiP`, so this
    test should load its table without issues.
    """
    M = get_M(formula="CrNiP")
    assert M(400) == 0.002092
    assert M(450) == 0.5 * (0.002092 + 0.002304)


def test_NdFe14B():
    """Test material `NdFe14B`.

    There is no material with such formula in the database,
    so we expect a `LookupError`.
    """
    with pytest.raises(LookupError):
        get_M(formula="NdFe14B")


def test_CrNiP_P1():
    """Test material `CrNiP` with space group name `P1`.

    There is no material with such formula and space group
    in the database, so we expect a `LookupError`.
    """
    with pytest.raises(LookupError):
        get_M(formula="Co2Fe2H4", space_group_name="P1")


def test_all():
    """Test search with no filters.

    This will select all entries in the database,
    so we expect a `LookupError`.
    """
    with pytest.raises(LookupError):
        get_M()


def test_uppasd_known():
    """Test query with UppASD input files.

    We expect this test to pass and retrieve a known material.
    """
    M = get_M(
        jfile=DATA / "known_material" / "jfile",
        momfile=DATA / "known_material" / "momfile",
        posfile=DATA / "known_material" / "posfile",
    )
    assert M(400) == 1.38120701
    assert M(450) == 0.5 * (1.3495301 + 1.33638686)


def test_uppasd_unknown():
    """Test query with UppASD input files.

    We expect this test to fail with a `LookupError`.
    """
    with pytest.raises(LookupError):
        get_M(
            jfile=DATA / "unknown_material" / "jfile",
            momfile=DATA / "unknown_material" / "momfile",
            posfile=DATA / "unknown_material" / "posfile",
        )


def test_uppasd_incorrect():
    """Test query with UppASD input files.

    We expect this test to fail with a `SyntaxError`.
    """
    with pytest.raises(SyntaxError):
        get_M(
            jfile=DATA / "wrong_data" / "jfile",
            momfile=DATA / "wrong_data" / "momfile",
            posfile=DATA / "wrong_data" / "posfile",
        )
