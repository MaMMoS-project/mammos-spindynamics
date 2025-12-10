"""UppASD Data class."""

from __future__ import annotations

import pathlib
import re
from typing import TYPE_CHECKING

import mammos_entity as me
import mammos_units as u
import numpy as np
import pandas as pd
import yaml

from mammos_spindynamics.uppasd.inpsd import parse_inpsd_file

if TYPE_CHECKING:
    import mammos_entity
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
        if info["metadata"]["mode"] == "mammos_uppasd_data":
            return MammosUppasdData(out)
        elif info["metadata"]["mode"] == "run":
            return RunData(out)
        elif info["metadata"]["mode"] == "temperature_sweep":
            return TemperatureSweepData(out)
        else:
            mode = info["metadata"]["mode"]
            raise RuntimeError(f"Cannot understand mode {mode} in info.yaml.")
    else:
        raise RuntimeError(f"`info.yaml` not found in path: {out}.")


class MammosUppasdData:
    """Collection of UppASD Result instances."""

    def __init__(self, out: pathlib.Path):
        """Initialize MammosUppasdData given the directory containing all runs."""
        self.out = pathlib.Path(out)
        runs = []
        for dir_ in self.out.iterdir():
            if re.match(r"(run|temperature_sweep)-\d+", dir_.name):
                with open(dir_ / "info.yaml") as file:
                    info = yaml.safe_load(file)
                runs.append({"id": dir_.name, **info["metadata"], **info["parameters"]})
        self._runs = runs

    def info(
        self,
        include_index: bool = False,
        include_description: bool = True,
        include_elapsed_time: bool = True,
        include_options: bool = True,
    ) -> pandas.DataFrame:
        """Dataframe containing information about all available UppASD runs."""
        keys = ["id"]
        if include_description:
            keys += ["description"]
        if include_elapsed_time:
            keys += ["time_elapsed"]
        if include_index:
            keys += ["index"]
        if include_options:
            keys += ["T"]
        return pd.DataFrame([{k: run[k] for k in keys} for run in self._runs])

    def __getitem__(self, idx):
        """Extract i-th run."""
        if self._runs[idx]["mode"] == "run":
            return RunData(self.out / self._runs[idx]["id"])
        elif self._runs[idx]["mode"] == "temperature_sweep":
            return TemperatureSweepData(self.out / self._runs[idx]["id"])
        else:
            raise RuntimeError("Runner not recognized.")

    def __repr__(self):
        """Define repr."""
        return f"MammosUppasdData('{self.out}')"


class RunData:
    """UppASD Data parser class for a single run."""

    def __init__(self, out: pathlib.Path):
        """Initialize Data given the output directory of a single run."""
        self.out = pathlib.Path(out)
        with open(self.out / "info.yaml") as f:
            info = yaml.safe_load(f)
        self.metadata = info["metadata"]
        self.parameters = info["parameters"]
        df = pd.read_csv(self.last_cumulant, sep=r"\s+", dtype=object)
        self.data = df.iloc[-1]
        self.input_dictionary = parse_inpsd_file(self.inpsd)

    def __repr__(self):
        """Define repr."""
        return f"RunData('{self.out}')"

    def describe(
        self,
        include_index: bool = False,
        include_description: bool = True,
        include_elapsed_time: bool = True,
        include_parameters: bool = True,
    ) -> pandas.DataFrame:
        """Return information about the UppASD run."""
        out = {"id": self.out.name}
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
        """Get path of ``jfile`` exchange file."""
        return pathlib.Path(self.input_dictionary["exchange"])

    @property
    def momfile(self) -> pathlib.Path:
        """Get path of ``momfile`` file."""
        return pathlib.Path(self.input_dictionary["momfile"])

    @property
    def posfile(self) -> pathlib.Path:
        """Get path of ``posfile`` file."""
        return pathlib.Path(self.input_dictionary["posfile"])

    @property
    def last_cumulant(self) -> str:
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
            return cumulant_files[-1]

    @property
    def restart_file(self) -> str:
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
        Ms_mu_B_per_atom = float(self.data["<M>"]) * u.mu_B
        Ms = Ms_mu_B_per_atom * self.n_magnetic_atoms / cell_volume
        return me.Ms(Ms, unit="A/m")

    @property
    def Cv(self) -> mammos_entity.Entity:
        """Get specific heat capacity."""
        Cv = float(self.data["C_v(tot)"])
        return me.Entity("IsochoricHeatCapacity", Cv)

    @property
    def U_binder(self) -> float:
        """Get U_binder value."""
        U_b = float(self.data["U_{Binder}"])
        return U_b


class TemperatureSweepData:
    """Class for the result of the temperature_array runner."""

    def __init__(self, out: pathlib.Path | str):
        """Initialize TemperatureSweepData given the output directory of a sweep run."""
        self.out = pathlib.Path(out)
        with open(self.out / "info.yaml") as f:
            info = yaml.safe_load(f)
        self.metadata = info["metadata"]
        self.parameters = info["parameters"]
        sub_runs = []
        for dir_ in self.out.iterdir():
            if re.match(r"run-\d+", dir_.name):
                with open(dir_ / "info.yaml") as f:
                    info = yaml.safe_load(f)
                sub_runs.append(
                    {"id": dir_.name, **info["metadata"], **info["parameters"]}
                )
        self._sub_runs = sub_runs

    def __repr__(self):
        """Define repr."""
        return f"TemperatureSweepData('{self.out}')"

    def __getitem__(self, idx):
        """Extract i-th sub-run."""
        return RunData(self.out / self._sub_runs[idx]["id"])

    def sel(self, **kwargs):
        """Select run satisfying certain filters defined in the keyword arguments."""
        df = self.describe()
        for key, val in kwargs.items():
            df = df[df[key] == val]
        label = df.iloc[0].id
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
        """Get Isochoric Heat Capatic."""
        return me.concat_flat(*[run.Cv for run in self])

    @property
    def U_binder(self) -> numpy.ndarray:
        """Get Binder coefficient."""
        return np.array([run.U_binder for run in self])

    # TODO: improved Cv
    # @property
    # def Cv(self) -> mammos_entity.Entity:
    #     """Get Specific Heat Capacity."""
    #     k_B = u.constants.k_B.to("mRy/K")  # Boltzmann constant in [mRy/K]
    #     E = missing...
    #     Cv = np.gradient(self.dataframe["<E>"] / k_B, self.dataframe["T"], axis=0)
    #     return me.Entity("IsochoricHeatCapacity", Cv)

    def describe(
        self,
        include_index: bool = False,
        include_description: bool = True,
        include_elapsed_time: bool = True,
        include_options: bool = True,
    ) -> pandas.DataFrame:
        """Dataframe containing information about all available UppASD runs."""
        keys = ["id"]
        if include_description:
            keys += ["description"]
        if include_elapsed_time:
            keys += ["time_elapsed"]
        if include_index:
            keys += ["index"]
        if include_options:
            keys += ["T"]
        return pd.DataFrame([{k: run[k] for k in keys} for run in self._sub_runs])

    # TODO: fix?
    # @property
    # def dataframe(self) -> pandas.DataFrame:
    #     """Dataframe containing information of the temperature_sweep."""
    #     list_data = []
    #     for sub_run in self._sub_runs:
    #         list_data.append(pd.concat([pd.Series({"T": sub_run.T}), sub_run.info]))
    #     df = pd.DataFrame(list_data).astype(
    #         {
    #             "T": "Int64",
    #             "#Iter": "Int64",
    #             "<M>": "Float64",
    #             "<M^2>": "Float64",
    #             "<M^4>": "Float64",
    #             "U_{Binder}": "Float64",
    #             r"\chi": "Float64",
    #             "C_v(tot)": "Float64",
    #             "<E>": "Float64",
    #             "<E_{exc}>": "Float64",
    #             "<E_{lsf}>": "Float64",
    #         }
    #     )
    #     return df

    def save_output(self, out: pathlib.Path | str | None = None) -> None:
        """Save output files M(T) and output.csv.

        The generated files are `M(T)` and `output.csv`.
        The first one contains all information evaluated from the cumulant files.
        The file `output.csv` is instead generated from :py:mod:`mammos_entity.io` using
        information from `M(T)` and converting all quantities to ontology units.
        """
        if out is None:
            out = self.out
        out = pathlib.Path(out)
        # np.savetxt(
        #     out / "M(T)",
        #     self.dataframe.to_numpy(),
        #     fmt=["%04d"] * 2 + ["% .8E"] * (self.dataframe.shape[1] - 2),
        #     header="T[K] iter"
        #     + "".join([f"{h:^16}" for h in self.dataframe.columns[2:]]),
        #     comments="",
        # )
        # TODO: just stack cumulant lines?
        me.io.entities_to_file(
            out / "output.csv",
            "Magnetization and heat capacity from UppASD",
            T=self.T,
            Ms=self.Ms,
            U_binder=self.dataframe["U_{Binder}"].to_numpy(),
            Cv=self.Cv,
        )
