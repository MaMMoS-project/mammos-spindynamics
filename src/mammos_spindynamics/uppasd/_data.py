"""UppASD Data class."""

from __future__ import annotations

import pathlib
import re
from io import StringIO
from typing import TYPE_CHECKING

import mammos_entity as me
import mammos_units as u
import numpy as np
import pandas as pd
import yaml

if TYPE_CHECKING:
    import mammos_entity
    import numpy
    import pandas


def read(out: pathlib.Path | str) -> MammosUppasdData | RunData | TemperatureSweepData:
    """Read UppASD calculations results directory."""
    out = pathlib.Path(out)
    if not out.is_dir():
        raise RuntimeError(f"Output directory {out} does not exist.")

    if (info_yaml := out / "mammos_spindynamics.yaml").is_file():
        with open(info_yaml) as f:
            info = yaml.safe_load(f)
        if info["metadata"]["mode"] == "mammos_uppasd_data":
            return MammosUppasdData(out)
        elif info["metadata"]["mode"] == "run":
            return RunData(out)
        elif info["metadata"]["mode"] == "temperature_sweep":
            return TemperatureSweepData(out)
        else:
            mode = info["metadata"]["mode"]
            raise RuntimeError(
                f"Unable to understand mode: '{mode}' in mammos_spindynamics.yaml. "
            )
    else:
        raise RuntimeError(f"mammos_spindynamics.yaml file not found in path: {out}.")


class MammosUppasdData:
    """Collection of UppASD Result instances."""

    def __init__(self, out: pathlib.Path):
        """Initialize MammosUppasdData given the directory containing all runs."""
        self.out = pathlib.Path(out)
        with open(self.out / "mammos_spindynamics.yaml") as f:
            info = yaml.safe_load(f)
        self.history = info["history"]

    def __repr__(self):
        """Define repr."""
        return f"MammosUppasdData('{self.out}')"

    def __len__(self):
        """Extract number of sub-runs."""
        return len(self.history)

    def __getitem__(self, idx):
        """Extract i-th run."""
        if idx < 0 or idx >= len(self):
            raise IndexError
        return read(self.out / self.history[idx])

    def info(
        self,
        include_index: bool = False,
        include_description: bool = True,
        include_elapsed_time: bool = True,
        include_parameters: bool = True,
    ) -> pandas.DataFrame:
        """Dataframe containing information about all available UppASD runs."""
        metadata_keys = []
        if include_description:
            metadata_keys += ["description"]
        if include_elapsed_time:
            metadata_keys += ["time_elapsed"]
        if include_index:
            metadata_keys += ["index"]
        all_runs = []
        for run in self:
            info_ = {"name": run.out.name}
            if metadata_keys:
                metadata_ = {k: run.metadata[k] for k in metadata_keys}
                info_ = {**info_, **metadata_}
            if include_parameters:
                info_ = {**info_, **run.parameters}
            all_runs.append(info_)
        return pd.DataFrame(all_runs)


class RunData:
    """UppASD Data parser class for a single run."""

    def __init__(self, out: pathlib.Path):
        """Initialize Data given the output directory of a single run."""
        self.out = pathlib.Path(out)
        with open(self.out / "mammos_spindynamics.yaml") as f:
            info = yaml.safe_load(f)
        self.metadata = info["metadata"]
        self.parameters = info["parameters"]
        df = pd.read_csv(self.last_cumulant, sep=r"\s+", dtype=object)
        self._cumulant_data = df.iloc[-1]
        self.input_dictionary = _parse_inpsd_file(self.inpsd)

    def __repr__(self):
        """Define repr."""
        return f"RunData('{self.out}')"

    def info(
        self,
        include_index: bool = False,
        include_description: bool = True,
        include_elapsed_time: bool = True,
        include_parameters: bool = True,
    ) -> pandas.DataFrame:
        """Return information about the UppASD run."""
        out = {"name": self.out.name}
        if include_description:
            out["description"] = self.metadata["description"]
        if include_elapsed_time:
            out["time_elapsed"] = self.metadata["time_elapsed"]
        if include_index:
            out["index"] = self.metadata["index"]
        if include_parameters:
            out = {**out, **self.parameters}
        return out

    @property
    def T(self) -> float:
        """Get Thermodynamics Temperature."""
        return me.T(self.input_dictionary["temp"])

    @property
    def inpsd(self) -> pathlib.Path:
        """Get path of ``inpsd.dat`` input file."""
        return pathlib.Path(self.out / "inpsd.dat")

    @property
    def exchange(self) -> pathlib.Path:
        """Get path of file containing exchange interactions."""
        return self.out / self.input_dictionary["exchange"]

    @property
    def momfile(self) -> pathlib.Path:
        """Get path of file containing magnetic moments."""
        return self.out / self.input_dictionary["momfile"]

    @property
    def posfile(self) -> pathlib.Path:
        """Get path of file containing atomic positions."""
        return self.out / self.input_dictionary["posfile"]

    @property
    def last_cumulant(self) -> pathlib.Path:
        """Get last ``cumulant*.out`` file."""
        cumulant_files = [
            f for f in self.out.iterdir() if re.match("cumulant.*.out", f.name)
        ]
        if not cumulant_files:
            raise ValueError(
                f"No cumulant files found in the output directory  {self.out}."
            )
        else:
            cumulant_files.sort()
            return pathlib.Path(cumulant_files[-1])

    @property
    def restartfile(self) -> pathlib.Path:
        """Get path of restart file."""
        name = self.input_dictionary.get("simid", "_UppASD_")
        return self.out / f"restart.{name}.out"

    @property
    def n_magnetic_atoms(self) -> int:
        """Get number of magnetic atoms."""
        with open(self.momfile, encoding="utf-8") as file:
            n = len(file.readlines())
        return n

    @property
    def Ms(self) -> mammos_entity.Entity:
        """Get Spontaneous Magnetization."""
        cell = self.input_dictionary["cell"]
        lattice_const = self.input_dictionary["alat"] * u.m
        cell_volume = np.dot(cell[0], np.cross(cell[1], cell[2])) * lattice_const**3
        Ms_mu_B_per_atom = float(self._cumulant_data["<M>"]) * u.mu_B
        Ms = Ms_mu_B_per_atom * self.n_magnetic_atoms / cell_volume
        return me.Ms(Ms, unit="A/m")

    @property
    def Cv(self) -> mammos_entity.Entity:
        """Get specific heat capacity."""
        k_B = u.constants.k_B
        Cv = float(self._cumulant_data["C_v(tot)"]) * k_B * self.n_magnetic_atoms
        return me.Entity("IsochoricHeatCapacity", Cv)

    @property
    def E(self) -> mammos_entity.Entity:
        """Get energy."""
        E = float(self._cumulant_data["<E>"]) * u.mRy * self.n_magnetic_atoms
        return me.Entity("Energy", E, unit="J")

    @property
    def U_binder(self) -> float:
        """Get U_binder value."""
        U_b = float(self._cumulant_data["U_{Binder}"])
        return U_b


class TemperatureSweepData:
    """Class for the result of the temperature_array runner."""

    def __init__(self, out: pathlib.Path | str):
        """Initialize TemperatureSweepData given the output directory of a sweep run."""
        self.out = pathlib.Path(out)
        with open(self.out / "mammos_spindynamics.yaml") as f:
            info = yaml.safe_load(f)
        self.metadata = info["metadata"]
        self.parameters = info["parameters"]

    def __repr__(self):
        """Define repr."""
        return f"TemperatureSweepData('{self.out}')"

    def __len__(self):
        """Extract number of sub-runs."""
        return len(self.parameters["T"])

    def __getitem__(self, idx):
        """Extract i-th sub-run."""
        if idx < 0 or idx >= len(self):
            raise IndexError
        return RunData(self.out / f"run-{idx}")

    def info(
        self,
        include_index: bool = False,
        include_description: bool = True,
        include_elapsed_time: bool = True,
        include_parameters: bool = True,
    ) -> pandas.DataFrame:
        """Dataframe containing information about all available UppASD runs."""
        metadata_keys = []
        if include_description:
            metadata_keys += ["description"]
        if include_elapsed_time:
            metadata_keys += ["time_elapsed"]
        if include_index:
            metadata_keys += ["index"]
        all_runs = []
        for run in self:
            info_ = {"name": run.out.name}
            if metadata_keys:
                metadata_ = {k: run.metadata[k] for k in metadata_keys}
                info_ = {**info_, **metadata_}
            if include_parameters:
                info_ = {**info_, **run.parameters}
            all_runs.append(info_)
        return pd.DataFrame(all_runs)

    def sel(self, **kwargs):
        """Select run satisfying certain filters defined in the keyword arguments."""
        df = self.info()
        for key, val in kwargs.items():
            df = df[df[key] == val]
        label = df.iloc[0]["name"]
        return RunData(self.out / label)

    @property
    def T(self) -> float:
        """Get Thermodynamics Temperature."""
        return me.concat_flat(*[run.T for run in self])

    @property
    def Ms(self) -> mammos_entity.Entity:
        """Get Spontaneous Magnetization."""
        return me.concat_flat(*[run.Ms for run in self])

    @property
    def Cv(self) -> mammos_entity.Entity:
        """Get Isochoric Heat Capacity.

        The isochorich (at constant volume) heat capacity Cv is evaluated as the
        derivative of the energy as a function of temperature.
        """
        k_B = u.constants.k_B.value
        Cv = np.gradient(self.E.value / k_B, self.T.value, axis=0)
        return me.Entity("IsochoricHeatCapacity", Cv)

    @property
    def U_binder(self) -> numpy.ndarray:
        """Get Binder coefficient."""
        return np.array([run.U_binder for run in self])

    @property
    def E(self) -> mammos_entity.Entity:
        """Get Energy."""
        return me.concat_flat(*[run.E for run in self])

    def save_output(self, out: pathlib.Path | str | None = None) -> None:
        """Save output file output.csv in directory `out`.

        The file `output.csv` is generated from :py:mod:`mammos_entity.io` and includes
        information like temperature, magnetization, Binder cumulant and heat capacity.
        """
        if out is None:
            out = self.out
        out = pathlib.Path(out)
        out.mkdir(parents=True, exist_ok=True)

        with open(self[0].last_cumulant) as f:
            lines = f.readlines()
        header = lines[0]

        with open(out / "M(T)", "w") as f:
            f.write(f"{'T':>5} {header}")
            for run in self:
                with open(run.last_cumulant) as f_run:
                    lines = f_run.readlines()
                f.write(f"{run.T.value:>5.0f} {lines[-1]}")

        me.io.entities_to_file(
            out / "output.csv",
            "Magnetization and heat capacity from UppASD",
            T=self.T,
            Ms=self.Ms,
            U_binder=self.U_binder,
            Cv=self.Cv,
            # E=self.E,
        )


def _parse_inpsd_file(inpsd_file: pathlib.Path | str) -> dict:
    """Parse inpsd.dat file."""
    string_parameters = {
        "simid",
        "exchange",
        "momfile",
        "posfile",
        "restartfile",
    }
    float_parameters = {
        "alat",
        "temp",
    }
    with open(inpsd_file, encoding="utf-8") as file:
        lines = file.readlines()
    parameters = {}
    for i, line in enumerate(lines):
        for par in string_parameters:
            if re.match(f"{par} .*", line):
                parameters[par] = line.split()[1]
        for par in float_parameters:
            if re.match(f"{par} .*", line):
                try:
                    parameters[par] = float(line.split()[1])
                except ValueError:
                    continue
        if re.match("(bc|BC) .*", line):
            parameters["bc"] = line.removeprefix("bc").split()[:3]
        if re.match("cell .*", line):
            a = np.genfromtxt(StringIO(line.removeprefix("cell")))
            b = np.genfromtxt(StringIO(lines[i + 1]))
            c = np.genfromtxt(StringIO(lines[i + 2]))
            parameters["cell"] = np.vstack((a, b, c))
        if re.match("ncell .*", line):
            parameters["ncell"] = np.genfromtxt(StringIO(line.removeprefix("ncell")))[
                :3
            ]
    return parameters
