"""Test uppasd interface."""

import pathlib

import mammos_entity as me
import numpy as np
import pandas as pd
import yaml

from mammos_spindynamics import uppasd


def test_read_function(DATA):
    """Test `read` function.

    The `mammos_spindynamics.uppasd.read` function should understand if the
    given input defines a `MammosUppasdData`, a `RunData`, or a `TemperatureSweepData`.
    """
    mammos_uppasd_data = uppasd.read(DATA / "uppasd")
    assert isinstance(mammos_uppasd_data, uppasd.MammosUppasdData)
    run_data = uppasd.read(DATA / "uppasd" / "1-run")
    assert isinstance(run_data, uppasd.RunData)
    temperature_sweep_data = uppasd.read(DATA / "uppasd" / "0-temperature_sweep")
    assert isinstance(temperature_sweep_data, uppasd.TemperatureSweepData)
    sub_run_data = uppasd.read(DATA / "uppasd" / "0-temperature_sweep" / "0-run")
    assert isinstance(sub_run_data, uppasd.RunData)


def test_MammosUppasdData_class(DATA):
    """Test MammosUppasdData class."""
    mammos_uppasd_data = uppasd.read(DATA / "uppasd")
    assert mammos_uppasd_data.out == pathlib.Path(DATA / "uppasd")
    info_df = pd.DataFrame(
        {
            "name": ["0-temperature_sweep", "1-run"],
            "description": ["Test sweep.", "Single run."],
            "time_elapsed": [None, "1 day, 1:01:01"],
            "T": [[2, 5], 10],
        },
        index=[0, 1],
    )
    assert info_df.equals(mammos_uppasd_data.info())
    assert isinstance(mammos_uppasd_data[0], uppasd.TemperatureSweepData)
    assert isinstance(mammos_uppasd_data[1], uppasd.RunData)


def test_MammosUppasdData_yaml(DATA):
    """Test MammosUppasdData yaml file."""
    with open(DATA / "uppasd" / "mammos_spindynamics.yaml") as f:
        info = yaml.safe_load(f)
    assert info == {
        "metadata": {"mode": "mammos_uppasd_data"},
    }


def test_RunData_class(DATA):
    """Test RunData class."""
    run_dir = DATA / "uppasd" / "1-run"
    run_data = uppasd.read(run_dir)
    assert run_data.out == pathlib.Path(run_dir)
    assert run_data.metadata == {
        "index": 1,
        "description": "Single run.",
        "UppASD version": "v6.0.2",
        "mammos_spindynamics_version": "0.2.5",
        "mode": "run",
        "time_elapsed": "1 day, 1:01:01",
        "time_end": "2025-03-15T13:04:26+01:00",
        "time_start": "2025-03-14T12:03:25+01:00",
    }
    assert run_data.parameters == {"T": 10}

    info_df = pd.DataFrame(
        {
            "name": ["1-run"],
            "description": ["Single run."],
            "time_elapsed": ["1 day, 1:01:01"],
            "T": [10],
        }
    )
    assert info_df.equals(run_data.info())
    assert me.T(10) == run_data.T
    assert me.Ms(6834.473593675746) == run_data.Ms
    assert me.Entity("IsochoricHeatCapacity", 0) == run_data.Cv
    assert me.Entity("Energy", 0) == run_data.E
    assert run_data.U_binder == 0.512274383
    assert run_data.inpsd == pathlib.Path(run_dir / "inpsd.dat")
    assert run_data.exchange == pathlib.Path(run_dir / "jfile")
    assert run_data.momfile == pathlib.Path(run_dir / "momfile")
    assert run_data.posfile == pathlib.Path(run_dir / "posfile")
    assert run_data.cumulants == pathlib.Path(run_dir / "cumulants.bccFe100.out")
    assert run_data.restartfile == pathlib.Path(run_dir / "restart.bccFe100.out")
    assert run_data.n_magnetic_atoms == 9


def test_RunData_yaml(DATA):
    """Test RunData yaml file."""
    with open(DATA / "uppasd" / "1-run" / "mammos_spindynamics.yaml") as f:
        info = yaml.safe_load(f)
    assert info == {
        "metadata": {
            "index": 1,
            "description": "Single run.",
            "UppASD version": "v6.0.2",
            "mammos_spindynamics_version": "0.2.5",
            "mode": "run",
            "time_elapsed": "1 day, 1:01:01",
            "time_end": "2025-03-15T13:04:26+01:00",
            "time_start": "2025-03-14T12:03:25+01:00",
        },
        "parameters": {"T": 10},
    }


def test_TemperatureSweepData_class(DATA):
    """Test TemperatureSweepData class."""
    sweep_dir = DATA / "uppasd" / "0-temperature_sweep"
    sweep_data = uppasd.read(sweep_dir)
    assert sweep_data.out == pathlib.Path(sweep_dir)
    assert sweep_data.metadata == {
        "index": 0,
        "description": "Test sweep.",
        "UppASD version": "v6.0.2",
        "mammos_spindynamics_version": "0.2.5",
        "mode": "temperature_sweep",
    }
    assert sweep_data.parameters == {"T": [2, 5]}

    info_df = pd.DataFrame(
        {
            "name": ["0-run", "1-run"],
            "description": ["", ""],
            "time_elapsed": ["20:28:52", "23 days, 0:30:43"],
            "T": [2, 5],
        },
        index=[0, 1],
    )
    assert info_df.equals(sweep_data.info())

    assert sweep_data.get(T=2).out == sweep_dir / "0-run"

    T = me.T([2, 5])
    Ms = me.Ms([6781.89022085, 6810.43736377])
    Cv = me.Entity("IsochoricHeatCapacity", [0, 0])
    E = me.Entity("Energy", [0, 0])
    U_binder = np.array([0.50682319, 0.50969701])
    assert T == sweep_data.T
    assert Ms == sweep_data.Ms
    assert Cv == sweep_data.Cv
    assert E == sweep_data.E
    assert np.allclose(U_binder, sweep_data.U_binder)


def test_TemperatureSweepData_output(DATA, tmp_path):
    sweep_data = uppasd.read(DATA / "uppasd" / "0-temperature_sweep")
    sweep_data.save_output(tmp_path)
    collection = me.io.entities_from_file(tmp_path / "output.csv")
    T = me.T([2, 5])
    Ms = me.Ms([6781.89022085, 6810.43736377])
    Cv = me.Entity("IsochoricHeatCapacity", [0, 0])
    U_binder = np.array([0.50682319, 0.50969701])
    E = me.Entity("Energy", [0, 0])
    assert collection.T == T
    assert collection.Ms == Ms
    assert collection.Cv == Cv
    assert np.allclose(collection.U_binder, U_binder)
    assert collection.E == E


def test_TemperatureSweepData_yaml(DATA):
    """Test TemperatureSweepData yaml file."""
    with open(
        DATA / "uppasd" / "0-temperature_sweep" / "mammos_spindynamics.yaml"
    ) as f:
        info = yaml.safe_load(f)
    assert info == {
        "metadata": {
            "index": 0,
            "description": "Test sweep.",
            "UppASD version": "v6.0.2",
            "mammos_spindynamics_version": "0.2.5",
            "mode": "temperature_sweep",
        },
        "parameters": {"T": [2, 5]},
    }
