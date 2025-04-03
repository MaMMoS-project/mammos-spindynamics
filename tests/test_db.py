"""Test db lookup."""

import pytest

from mammos_spindynamics.db import get_M


def test_CrNiP():
    """Test material `CrNiP`.

    There is only one material with formula `CrNiP`, so this
    test should load its table without issues.
    """
    M = get_M(formula="CrNiP")
    assert M(400) == 0.002092


def test_NdFe14B():
    """Test material `NdFe14B`.

    There is no material with such formula in the database,
    so we expect a `LookupError`.
    """
    with pytest.raises(LookupError):
        get_M(formula="NdFe14B")


def test_CrNiP_12345():
    """Test material `CrNiP` with structure `12345`.

    There is no material with such formula and structure
    in the database, so we expect a `LookupError`.
    """
    with pytest.raises(LookupError):
        get_M(formula="NdFe14B", structure="12345")
