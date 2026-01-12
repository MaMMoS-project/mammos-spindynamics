"""UppASD Data module.

This module finds, parse and operates on data generated from the UppASD software.

The :py:func:`~mammos_spindynamics.uppasd.read` function can be used to parse any of
three possible directories:

- :py:class:`~mammos_spindynamics.uppasd.MammosUppasdData` is the base directory where
  all sweeps and runs are found.
- :py:class:`~mammos_spindynamics.uppasd.TemperatureSweepData` is the output directory
  of a temperature sweep.
- :py:class:`~mammos_spindynamics.uppasd.RunData` is the output directory of a single
  run.

The :py:func:`~mammos_spindynamics.uppasd.read` function loads data lazily, only parsing
the type of the output directory initially.
"""

from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import mammos_entity as me
import mammos_units as u
import numpy as np
import pandas as pd
import yaml

if TYPE_CHECKING:
    import pathlib

    import mammos_entity
    import numpy
    import pandas


def read(
    out: pathlib.Path | str,
) -> MammosUppasdData | RunData | TemperatureSweepData:
    """Read UppASD calculations results directory.

    This function returns different classes based on the content of the
    `mammos_spindynamics.yaml` generated from the execution of UppASD via the
    `mammos_spindynamics` Python library.

    Args:
        out: Output directory to read.

    Returns:
        Relevant Data object based on the content of the given directory.

    """
    out = Path(out)
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
    """Collection of UppASD Result instances.

    This class collects all temperature sweeps and runs executed in the output directory
    specified by the `path` attribute.

    Attributes:
        out (pathlib.Path): Output directory containing temperature sweeps and
            runs.

    """

    def __init__(self, out: pathlib.Path | str):
        """Initialize MammosUppasdData given the directory containing all runs.

        Args:
            out: Output directory containing temperature sweeps and runs.

        Examples:
            >>> from mammos_spindynamics.uppasd import MammosUppasdData
            >>> MammosUppasdData("Fe3Y")
            MammosUppasdData('Fe3Y')
        """
        self.out = Path(out)

    def __repr__(self):
        """Define repr."""
        return f"MammosUppasdData('{self.out}')"

    def __getitem__(self, idx):
        """Extract i-th run."""
        runs = [
            run_dir
            for run_dir in self.out.iterdir()
            if re.match(r"\d+-(run|temperature_sweep)$", run_dir.name)
        ]
        runs.sort(key=lambda x: int(x.name.split("-")[0]))
        return read(runs[idx])

    def get(self, **kwargs) -> RunData | TemperatureSweepData:
        """Select run based on different filters specified as keyword arguments.

        Exceptions are raised if the filters give no match or too many matches.

        Args:
            **kwargs: Keyword arguments.

        Returns:
            :py:class:`~mammos_spindynamics.uppasd.RunData` or
            :py:class:`~mammos_spindynamics.uppasd.TemperatureSweepData` satisfying
            given filters.

        Raises:
            LookupError: No run has been found satisfying all filters.
            LookupError: Too many results have been found.

        """
        df = self.info()
        for key, value in kwargs.items():
            df = df[df[key] == value]
        if len(df) == 0:
            raise LookupError(
                f"Requested run not found.\nSelection: {kwargs}\n"
                f"Available data: {self.info()}"
            )
        if len(df) > 1:
            error_string = (
                "Too many results. Please refine your search.\n"
                + "Avilable materials based on request:\n"
            )
            error_string += str(df)
            raise LookupError(error_string)
        return read(self.out / df.iloc[0]["name"])

    def info(
        self,
        include_description: bool = True,
        include_time_elapsed: bool = True,
        include_parameters: bool = True,
    ) -> pandas.DataFrame:
        """Return a Dataframe containing information about all available UppASD runs.

        Args:
            include_description: Whether to include the description in the dataframe
                columns.
            include_time_elapsed: Whether to include elapsed time in the dataframe
                columns.
            include_parameters: Whether to include the run parameters in the dataframe
                columns.

        Returns:
            Dataframe with relevant information.
        """
        metadata_keys = []
        if include_description:
            metadata_keys += ["description"]
        if include_time_elapsed:
            metadata_keys += ["time_elapsed"]
        all_runs = []
        index = []
        for run in self:
            if run:
                info_ = {"name": run.out.name}
                index.append(int(run.out.name.split("-")[0]))
                if metadata_keys:
                    metadata_ = {k: run.metadata.get(k) for k in metadata_keys}
                    info_ = {**info_, **metadata_}
                if include_parameters:
                    info_ = {**info_, **run.parameters}
                all_runs.append(info_)
        return pd.DataFrame(all_runs, index=index)


class RunData:
    """UppASD Data parser class for a single run.

    Attributes:
        out (pathlib.Path): Output directory containing the run output.
        metadata (dict): Dictionary of the run metadata.
        parameters (dict): Dictionary of the run parameters.
    """

    def __init__(self, out: pathlib.Path):
        """Initialize RunData given the directory path of a single run.

        Args:
            out: Output directory containing the run output.
        """
        self.out = Path(out)
        with open(self.out / "mammos_spindynamics.yaml") as f:
            info = yaml.safe_load(f)
        self._input_dictionary = _parse_inpsd_file(self.inpsd)
        self.metadata = info["metadata"]
        self.parameters = info["parameters"]
        df = pd.read_csv(self.cumulants, sep=r"\s+")
        self._cumulant_data = df.iloc[-1]

    def __repr__(self):
        """Define repr."""
        return f"RunData('{self.out}')"

    def info(
        self,
        include_description: bool = True,
        include_time_elapsed: bool = True,
        include_parameters: bool = True,
    ) -> pandas.DataFrame:
        """Return a Dataframe containing information about the UppASD run.

        Args:
            include_description: Whether to include the description in the dataframe
                columns.
            include_time_elapsed: Whether to include elapsed time in the dataframe
                columns.
            include_parameters: Whether to include the run parameters in the dataframe
                columns.

        Returns:
            Dataframe with relevant information.
        """
        out = {"name": [self.out.name]}
        if include_description:
            out["description"] = [self.metadata["description"]]
        if include_time_elapsed:
            out["time_elapsed"] = [self.metadata["time_elapsed"]]
        if include_parameters:
            parameters_dictionary = {k: [v] for k, v in self.parameters.items()}
            out = {**out, **parameters_dictionary}
        return pd.DataFrame(out)

    @property
    def T(self) -> mammos_entity.Entity:
        """Get temperature of the run.

        Returns:
            Entity Thermodynamic Temperature in Kelvin.
        """
        return me.T(self._input_dictionary["temp"])

    @property
    def inpsd(self) -> pathlib.Path:
        """Get path of ``inpsd.dat`` input file.

        Returns:
            Path of input file defining simulation parameters.
        """
        return Path(self.out / "inpsd.dat")

    @property
    def exchange(self) -> pathlib.Path:
        """Get path of file containing exchange interactions.

        Returns:
            Path of file defining exchange interactions.
        """
        return self.out / self._input_dictionary["exchange"]

    @property
    def momfile(self) -> pathlib.Path:
        """Get path of file containing magnetic moments.

        Returns:
            Path of file defining magnetic moments.
        """
        return self.out / self._input_dictionary["momfile"]

    @property
    def posfile(self) -> pathlib.Path:
        """Get path of file containing atomic positions.

        Returns:
            Path of file defining atomic positions.
        """
        return self.out / self._input_dictionary["posfile"]

    @property
    def cumulants(self) -> pathlib.Path:
        """Get ``cumulants.*.out`` file.

        Returns:
            Path of file containing magnetic information.

        Raises:
            RuntimeError: Could not find the cumulants file.
            RuntimeError: More than one cumulants file were found.
        """
        cumulants = list(self.out.glob("cumulants.*.out"))
        if not cumulants:
            raise RuntimeError(f"Could not find a cumulants.<simid>.out in {self.out}")
        if len(cumulants) > 1:
            # we assume that this will never happen
            raise RuntimeError(
                f"Found multiple candidates for cumulants.<simid>.out in {self.out}:\n"
                f"{cumulants}"
            )
        return cumulants[0]

    @property
    def restartfile(self) -> pathlib.Path:
        """Get path of restart file.

        Returns:
            Path of restart file for consecutive runs.

        Raises:
            RuntimeError: Could not find the restart file.
            RuntimeError: More than one restart file were found.
        """
        restart_files = list(self.out.glob("restart.*.out"))
        if not restart_files:
            raise RuntimeError(f"Could not find restart.<simid>.out in {self.out}")
        if len(restart_files) > 1:
            # we assume that this will never happen
            raise RuntimeError(
                f"Found multiple candidates for restart.<simid>.out in {self.out}:\n"
                f"{restart_files}"
            )
        return restart_files[0]

    @property
    def n_magnetic_atoms(self) -> int:
        """Get number of magnetic atoms.

        Returns:
            Number of magnetic atoms as defined in the ``momfile``.
        """
        with open(self.momfile, encoding="utf-8") as file:
            n = len(file.readlines())
        return n

    @property
    def Ms(self) -> mammos_entity.Entity:
        """Get spontaneous magnetization of the run.

        Returns:
            Entity Spontaneous Magnetization in Ampere per meter.
        """
        cell = self._input_dictionary["cell"]
        lattice_const = self._input_dictionary["alat"] * u.m
        cell_volume = np.dot(cell[0], np.cross(cell[1], cell[2])) * lattice_const**3
        Ms_mu_B_per_atom = float(self._cumulant_data["<M>"]) * u.mu_B
        Ms = Ms_mu_B_per_atom * self.n_magnetic_atoms / cell_volume
        return me.Ms(Ms, unit="kA/m")

    @property
    def Cv(self) -> mammos_entity.Entity:
        """Get specific heat capacity at constant volume.

        Returns:
            Entity IsochoricHeatCapacity in Joule per Kelvin.
        """
        k_B = u.constants.k_B
        Cv = float(self._cumulant_data["C_v(tot)"]) * k_B * self.n_magnetic_atoms
        return me.Entity("IsochoricHeatCapacity", Cv)

    @property
    def E(self) -> mammos_entity.Entity:
        """Get energy.

        Returns:
            Entity Energy in Joule.
        """
        E = float(self._cumulant_data["<E>"]) * u.mRy * self.n_magnetic_atoms
        return me.Entity("Energy", E, unit="J")

    @property
    def U_binder(self) -> float:
        """Get U_binder coefficient.

        Returns:
            Binder coefficient.
        """
        U_b = float(self._cumulant_data["U_{Binder}"])
        return U_b


class TemperatureSweepData:
    """UppASD Data parser for a temperature sweep.

    A temperature sweep is a sequence of runs with changing temperature and possibly
    other simulation parameters.

    Attributes:
        out (pathlib.Path): Output directory containing the sweep output.
        metadata (dict): Dictionary of the sweep metadata.
        parameters (dict): Dictionary of the sweep parameters.

    """

    def __init__(self, out: pathlib.Path | str):
        """Initialize TemperatureSweepData given the directory path of a sweep run.

        Args:
            out: Output directory containing the sweep output.
        """
        self.out = Path(out)
        with open(self.out / "mammos_spindynamics.yaml") as f:
            info = yaml.safe_load(f)
        self.metadata = info["metadata"]
        self.parameters = info["parameters"]

    def __repr__(self):
        """Define repr."""
        return f"TemperatureSweepData('{self.out}')"

    def __getitem__(self, idx):
        """Extract i-th run."""
        runs = [
            run_dir
            for run_dir in self.out.iterdir()
            if re.match(r"\d+-run", run_dir.name)
        ]
        runs.sort(key=lambda x: int(x.name.split("-")[0]))
        return read(runs[idx])

    def info(
        self,
        include_description: bool = True,
        include_time_elapsed: bool = True,
        include_parameters: bool = True,
    ) -> pandas.DataFrame:
        """Return a Dataframe containing information about the temperature sweep.

        Args:
            include_description: Whether to include the description in the dataframe
                columns.
            include_time_elapsed: Whether to include elapsed time in the dataframe
                columns.
            include_parameters: Whether to include the run parameters in the dataframe
                columns.

        Returns:
            Dataframe with relevant information.
        """
        metadata_keys = []
        if include_description:
            metadata_keys += ["description"]
        if include_time_elapsed:
            metadata_keys += ["time_elapsed"]
        all_runs = []
        index = []
        for run in self:
            if run:
                info_ = {"name": run.out.name}
                index.append(
                    int(
                        run.out.name.replace("-run", "").replace(
                            "-temperature_sweep", ""
                        )
                    )
                )
                if metadata_keys:
                    metadata_ = {k: run.metadata[k] for k in metadata_keys}
                    info_ = {**info_, **metadata_}
                if include_parameters:
                    info_ = {**info_, **run.parameters}
                all_runs.append(info_)
        return pd.DataFrame(all_runs, index=index)

    def get(self, **kwargs) -> RunData:
        """Select run based on different filters specified as keyword arguments.

        Exceptions are raised if the filters give no match or too many matches.

        Args:
            **kwargs: Keyword arguments.

        Returns:
            :py:class:`~mammos_spindynamics.uppasd.RunData` satisfying
            given filters.

        Raises:
            LookupError: No run has been found satisfying all filters.
            LookupError: Too many results have been found.

        """
        df = self.info()
        for key, val in kwargs.items():
            df = df[df[key] == val]
        if len(df) == 0:
            raise LookupError(
                f"Requested run not found.\nSelection: {kwargs}\n"
                f"Available data: {self.info()}"
            )
        if len(df) > 1:
            error_string = (
                "Too many results. Please refine your search.\n"
                + "Avilable materials based on request:\n"
            )
            error_string += str(df)
            raise LookupError(error_string)
        return read(self.out / df.iloc[0]["name"])

    @property
    def T(self) -> mammos_entity.Entity:
        """Get array of temperatures of the sweep.

        Returns:
            1D array Entity ThermodynamicTemperature in Kelvin.
        """
        return me.concat_flat(*[run.T for run in self if run])

    @property
    def Ms(self) -> mammos_entity.Entity:
        """Get array of spontaneous magnetization of the sweep.

        Returns:
            1D array Entity SpontaneousMagnetization in Kelvin.
        """
        return me.concat_flat(*[run.Ms for run in self if run])

    @property
    def Cv(self) -> mammos_entity.Entity:
        """Get specific heat capacity at constant volume.

        For a temperature sweep the heat capacity Cv is evaluated as the
        derivative of the energy as a function of temperature.

        Returns:
            1D array Entity IsochoricHeatCapacity in Joule per Kelvin.
        """
        k_B = u.constants.k_B.value
        Cv = np.gradient(self.E.value / k_B, self.T.value, axis=0)
        return me.Entity("IsochoricHeatCapacity", Cv)

    @property
    def U_binder(self) -> numpy.ndarray:
        """Get array of Binder coefficients.

        Returns:
            Array of Binder coefficients.
        """
        return np.array([run.U_binder for run in self if run])

    @property
    def E(self) -> mammos_entity.Entity:
        """Get energy.

        Returns:
            1D array Entity Energy in Joule.
        """
        return me.concat_flat(*[run.E for run in self if run])

    def save_output(self, out: pathlib.Path | str) -> None:
        """Save output files M(T) and output.csv in directory `out`.

        `M(T)` contains all information evaluated from the cumulant files.
        `output.csv` contains entities for information temperature,
        magnetization, Binder cumulant and heat capacity.

        Args:
            out: Directory location where to save output files.
        """
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)

        with open(self[0].cumulants) as f:
            lines = f.readlines()
        header = lines[0]

        with open(out / "M(T)", "w") as f:
            f.write(f"{'T':>5} {header}")
            for run in self:
                if run:
                    with open(run.cumulants) as f_run:
                        lines = f_run.readlines()
                    f.write(f"{run.T.value:>5.0f} {lines[-1]}")

        me.io.entities_to_file(
            out / "output.csv",
            "Magnetization and heat capacity from UppASD",
            T=self.T,
            Ms=self.Ms,
            U_binder=self.U_binder,
            Cv=self.Cv,
            E=self.E,
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
