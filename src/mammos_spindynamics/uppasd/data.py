"""UppASD Data class."""

from __future__ import annotations

import datetime
import fnmatch
import yaml
import pathlib
import shutil
import subprocess
from io import StringIO
from typing import TYPE_CHECKING, Any, Iterable

import mammos_entity as me
import mammos_units as u
import numpy as np
import pandas as pd
from pydantic import field_validator
from pydantic.dataclasses import dataclass

import mammos_spindynamics
from mammos_spindynamics.uppasd.inpsd import parse_inpsd_file

if TYPE_CHECKING:
    import mammos_entity
    import mammos_units
    import numpy
    import pandas


def read(out: pathlib.Path | str) -> MammosUppasdData | RunData | TemperatureSweepData:
    """Read UppASD calculations results directory."""
    out = pathlib.Path(out)
    if not out.is_dir():
        raise RuntimeError(f"Output directory {out} does not exist.")

    if (info_yaml := out / "info.yaml").is_file():
        with open(info_yaml) as f:
            info = yaml.safe_load(f)
        if info["metadata"]["mode"] == "run":
            return RunData(out)
        elif info["metadata"]["mode"] == "temperature_sweep":
            return TemperatureSweepData(out)
        else:
            mode = info["metadata"]["mode"]
            raise RuntimeError(f"Cannot understand mode {mode} in info.yaml.")
    else:
        return MammosUppasdData(out)



class MammosUppasdData:
    """Collection of UppASD Result instances."""

    def __init__(self, out: pathlib.Path):
        self.out = pathlib.Path(out)
        runs = []
        for d_ in self.out.iterdir():
            if fnmatch.fnmatch(d_.name, "run-[0-9]*") or fnmatch.fnmatch(
                d_.name, "temperature_sweep-[0-9]*"
            ):
                with open(d_ / "info.yaml") as file:
                    info = yaml.safe_load(file)
                runs.append({"id": d_.name, **info["metadata"], **info["parameters"]})
        self.runs = runs

    @property
    def info(self) -> pandas.DataFrame:
        """Dataframe containing information about all available UppASD runs."""
        return pd.DataFrame(self.runs)

    def __getitem__(self, idx):
        """Extract i-th run."""
        if self.runs[idx]["mode"] == "run":
            return RunData(self.out / self.runs[idx]["id"])
        elif self.runs[idx]["mode"] == "temperature_sweep":
            return TemperatureSweepData(self.out / self.runs[idx]["id"])
        else:
            raise RuntimeError("Runner not recognized.")

    def __repr__(self):
        return f"MammosUppasdData('{self.out}')"

    def _repr_html_(self):
        return self.info._repr_html_()


class RunData:
    """UppASD Data parser class for a single run."""

    def __init__(self, out: pathlib.Path):
        """Initialize Data given the output `run` directory."""
        self.out = pathlib.Path(out)
        with open(self.out / "info.yaml") as file:
            info = yaml.safe_load(file)
        self.metadata = info["metadata"]
        self.parameters = info["parameters"]
        df = pd.read_csv(self.last_cumulant, sep=r"\s+", dtype=object)
        self.data = df.iloc[-1]
        self.input_dictionary = parse_inpsd_file(self.inpsd)

    def __repr__(self):
        return f"RunData('{self.out}')"

    def describe(self, **kwargs):
        """Describe something."""
        # TODO
        return

    @property
    def T(self) -> float:
        """Return temperature."""
        return self.input_dictionary["temp"]

    @property
    def inpsd(self) -> str:
        """Return path of ``inpsd.dat`` file."""
        return self.out / "inpsd.dat"

    @property
    def jfile(self) -> str:
        """Return path of ``jfile`` file."""
        return self.out / "jfile"

    @property
    def momfile(self) -> str:
        """Return path of ``momfile`` file."""
        return self.out / "momfile"

    @property
    def posfile(self) -> str:
        """Return path of ``posfile`` file."""
        return self.out / "posfile"

    @property
    def last_cumulant(self) -> str:
        """Return last ``cumulant*.out`` file."""
        cumulant_files = [
            f
            for f in self.out.iterdir()
            if fnmatch.fnmatch(f.name, "cumulant*.out")
        ]
        if not cumulant_files:
            raise ValueError(
                f"No cumulant files found in the output directory  {self.out}."
            )
        else:
            cumulant_files.sort()
            return cumulant_files[-1]

    @property
    def restart_file(self) -> str:
        """Return path of restart file."""
        return self.out / f"restart.{self.input_dictionary['simid']}.out"

    @property
    def n_magnetic_atoms(self) -> int:
        """Evaluate number of magnetic atoms."""
        with open(self.momfile, encoding="utf-8") as file:
            n = len(file.readlines())
        return n

    @property
    def Ms(self) -> mammos_entity.Entity:
        """Evaluate Spontaneous Magnetization."""
        cell = self.input_dictionary["cell"]
        lattice_const = self.input_dictionary["alat"] * u.m
        cell_volume = np.dot(cell[0], np.cross(cell[1], cell[2])) * lattice_const ** 3
        Ms_mu_B_per_atom = float(self.data["<M>"]) * u.mu_B
        Ms = Ms_mu_B_per_atom * self.n_magnetic_atoms / cell_volume
        return me.Ms(Ms, unit="A/m")

    @property
    def Cv(self) -> mammos_entity.Entity:
        """Evaluate something."""
        # TODO
        pass


class TemperatureSweepData:
    """Class for the result of the temperature_array runner."""

    def __init__(self, run_dir: pathlib.Path | str):
        """Initialize TemperatureArrayResult given the run directory."""
        self.run_dir = pathlib.Path(run_dir)
        with open(self.run_dir / "info.yaml") as file:
            info = yaml.safe_load(file)
        self.temperature_array = info["temperature"]
        self.sub_runs = []
        for sub_run_dir in self.run_dir.iterdir():
            if "run_" in sub_run_dir.name:
                self.sub_runs.append(Result(sub_run_dir))

    @property
    def volume(self) -> me.Entity:
        """Evaluate cell volume."""
        return self.sub_runs[0].volume

    @property
    def n_magnetic_atoms(self) -> int:
        """Evaluate number of magnetic atoms."""
        return self.sub_runs[0].n_magnetic_atoms

    @property
    def dataframe(self) -> pandas.DataFrame:
        """Dataframe containing information of the temperature_array run."""
        list_data = []
        for sub_run in self.sub_runs:
            list_data.append(
                pd.concat(
                    [pd.Series({"temperature": sub_run.temperature}), sub_run.data]
                )
            )
        df = pd.DataFrame(list_data).astype(
            {
                "temperature": "Int64",
                "#Iter": "Int64",
                "<M>": "Float64",
                "<M^2>": "Float64",
                "<M^4>": "Float64",
                "U_{Binder}": "Float64",
                r"\chi": "Float64",
                "C_v(tot)": "Float64",
                "<E>": "Float64",
                "<E_{exc}>": "Float64",
                "<E_{lsf}>": "Float64",
            }
        )
        return df

    def save_output(self, out: pathlib.Path | str | None = None) -> None:
        """Save output files M(T) and output.csv.

        The generated files are `M(T)` and `output.csv`.
        The first one contains all information evaluated from the cumulant files.
        The file `output.csv` is instead generated from :py:mod:`mammos_entity.io` using
        information from `M(T)` and converting all quantities to ontology units.
        """
        if out is None:
            out = self.run_dir
        out = pathlib.Path(out)
        np.savetxt(
            out / "M(T)",
            self.dataframe.to_numpy(),
            fmt=["%04d"] * 2 + ["% .8E"] * (self.dataframe.shape[1] - 2),
            header="T[K] iter"
            + "".join([f"{h:^16}" for h in self.dataframe.columns[2:]]),
            comments="",
        )
        Ms_mu_B_per_atom = self.dataframe["<M>"].to_numpy() * u.mu_B
        Ms = Ms_mu_B_per_atom * self.n_magnetic_atoms / self.volume.q
        k_B = u.constants.k_B.to("mRy/K")  # Boltzmann constant in [mRy/K]
        Cv = np.gradient(
            self.dataframe["<E>"] / k_B, self.dataframe["temperature"], axis=0
        )
        me.io.entities_to_file(
            out / "output.csv",
            "Magnetization and heat capacity from UppASD",
            T=me.Entity(
                "ThermodynamicTemperature", self.dataframe["temperature"].to_numpy()
            ),
            Ms=me.Ms(Ms, unit="A/m"),
            U_binder=self.dataframe["U_{Binder}"].to_numpy(),
            Cv=me.Entity("IsochoricHeatCapacity", Cv),
        )
