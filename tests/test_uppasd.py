"""Test uppasd interface."""

import json

import mammos_units as u
import pytest
from pydantic import ValidationError

import mammos_spindynamics
from mammos_spindynamics import uppasd


def test_jfile_not_found():
    """Test simulation with jfile input file not found on system."""
    with pytest.raises(FileNotFoundError):
        uppasd.Simulation(
            jfile="does_not_exist",
        )


def test_jfile_not_given(DATA):
    """Test simulation with jfile input file not given."""
    with pytest.raises(ValidationError):
        uppasd.Simulation(
            momfile=DATA / "uppasd/momfile",
            posfile=DATA / "uppasd/posfile",
            inpsd=DATA / "uppasd/inpsd.dat",
        )


def test_momfile_not_found():
    """Test simulation with momfile input file not found on system."""
    with pytest.raises(FileNotFoundError):
        uppasd.Simulation(
            momfile="does_not_exist",
        )


def test_momfile_not_given(DATA):
    """Test simulation with momfile input file not given."""
    with pytest.raises(ValidationError):
        uppasd.Simulation(
            jfile=DATA / "uppasd/jfile",
            posfile=DATA / "uppasd/posfile",
            inpsd=DATA / "uppasd/inpsd.dat",
        )


def test_posfile_not_found():
    """Test simulation with posfile input file not found on system."""
    with pytest.raises(FileNotFoundError):
        uppasd.Simulation(
            posfile="does_not_exist",
        )


def test_posfile_not_given(DATA):
    """Test simulation with posfile input file not given."""
    with pytest.raises(ValidationError):
        uppasd.Simulation(
            jfile=DATA / "uppasd/jfile",
            momfile=DATA / "uppasd/momfile",
            inpsd=DATA / "uppasd/inpsd.dat",
        )


def test_inpsd_not_found():
    """Test simulation with inpsd input file not found on system."""
    with pytest.raises(FileNotFoundError):
        uppasd.Simulation(
            inpsd="does_not_exist",
        )


def test_inpsd_not_given(DATA):
    """Test simulation with inpsd input file not given."""
    with pytest.raises(ValidationError):
        uppasd.Simulation(
            jfile=DATA / "uppasd/jfile",
            momfile=DATA / "uppasd/momfile",
            posfile=DATA / "uppasd/posfile",
        )


def test_volume(DATA):
    """Test cell volume evaluation."""
    sim = uppasd.Simulation(
        jfile=DATA / "uppasd/jfile",
        momfile=DATA / "uppasd/momfile",
        posfile=DATA / "uppasd/posfile",
        inpsd=DATA / "uppasd/inpsd.dat",
    )
    assert sim.volume.q == 103.7906328202101 * u.AA**3


def test_n_magnetic_atoms(DATA):
    """Test evaluation of number of magnetic atoms."""
    sim = uppasd.Simulation(
        jfile=DATA / "uppasd/jfile",
        momfile=DATA / "uppasd/momfile",
        posfile=DATA / "uppasd/posfile",
        inpsd=DATA / "uppasd/inpsd.dat",
    )
    assert sim.n_magnetic_atoms == 9


def test_get_temperature_from_inpsd(DATA):
    """Test evaluation of temperature from inpsd.dat file."""
    sim = uppasd.Simulation(
        jfile=DATA / "uppasd/jfile",
        momfile=DATA / "uppasd/momfile",
        posfile=DATA / "uppasd/posfile",
        inpsd=DATA / "uppasd/inpsd.dat",
    )
    assert sim.get_temperature_from_inpsd() == 10


def test_run(DATA, tmp_path):
    """Test UppASD simulation with the `run` method."""
    sim = uppasd.Simulation(
        jfile=DATA / "uppasd/jfile",
        momfile=DATA / "uppasd/momfile",
        posfile=DATA / "uppasd/posfile",
        inpsd=DATA / "uppasd/inpsd.dat",
    )
    temperature = 20
    sim.run(
        temperature,
        out_dir=tmp_path,
    )

    results = uppasd.ResultCollection(tmp_path)
    res = results[0]
    run_info = res.info

    # Load generated info.json and check informations
    with open(tmp_path / "run_0" / "info.json") as file:
        info_json = json.load(file)
    for key, value in run_info.items():
        assert info_json[key] == value

    # Check simulation metadata
    assert run_info["runner"] == "run"
    assert run_info["mammos_spindynamics_version"] == mammos_spindynamics.__version__

    # Check correct temperature has been used
    assert res.temperature == temperature


# TODO: temperature array test?
