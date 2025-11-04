"""Functions for interfacing with UppASD."""

from __future__ import annotations

import datetime
import fnmatch
import json
import pathlib
import shutil
import subprocess
from io import StringIO
from typing import TYPE_CHECKING, Any

import mammos_entity as me
import mammos_units as u
import numpy as np
import pandas as pd
from pydantic import field_validator
from pydantic.dataclasses import dataclass

import mammos_spindynamics

if TYPE_CHECKING:
    import pandas

_uppasd_bin = shutil.which("sd.gfortran")


@dataclass()
class Simulation:
    """UppASD Simulation class."""

    jfile: pathlib.Path
    momfile: pathlib.Path
    posfile: pathlib.Path
    inpsd: pathlib.Path

    @field_validator("jfile", mode="before")
    @classmethod
    def _check_jfile(cls, jfile: Any) -> Any:
        """Check if jfile exists."""
        jfile = pathlib.Path(jfile)
        if not jfile.is_file():
            raise FileNotFoundError()
        return jfile

    @field_validator("momfile", mode="before")
    @classmethod
    def _check_momfile(cls, momfile: Any) -> Any:
        """Check if momfile exists."""
        momfile = pathlib.Path(momfile)
        if not momfile.is_file():
            raise FileNotFoundError()
        return momfile

    @field_validator("posfile", mode="before")
    @classmethod
    def _check_posfile(cls, posfile: Any) -> Any:
        """Check if posfile exists."""
        posfile = pathlib.Path(posfile)
        if not posfile.is_file():
            raise FileNotFoundError()
        return posfile

    @field_validator("inpsd", mode="before")
    @classmethod
    def _check_inpsd(cls, inpsd: Any) -> Any:
        """Check if inpsd.dat exists."""
        inpsd = pathlib.Path(inpsd)
        if not inpsd.is_file():
            raise FileNotFoundError()
        return inpsd

    @property
    def volume(self) -> me.Entity:
        """Evaluate cell volume."""
        with open(self.inpsd, encoding="utf-8") as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if fnmatch.fnmatch(line, "cell*"):
                    a = np.genfromtxt(StringIO(line[4:])) * u.Angstrom
                    b = np.genfromtxt(StringIO(lines[i + 1])) * u.Angstrom
                    c = np.genfromtxt(StringIO(lines[i + 2])) * u.Angstrom
                    V = me.Entity("CellVolume", np.dot(a, np.cross(b, c)))
                    break
            else:
                raise RuntimeError("Cell volume could not be evaluated.")
        return V

    @property
    def n_magnetic_atoms(self) -> int:
        """Evaluate number of magnetic atoms."""
        with open(self.momfile, encoding="utf-8") as file:
            n = len(file.readlines())
        return n

    @property
    def get_T_from_inpsd(self) -> float:
        """Read temperature from inpsd.dat file."""
        with open(self.inpsd, encoding="utf-8") as file:
            lines = file.readlines()
        for ll in lines:
            if fnmatch.fnmatch(ll, "temp *"):
                T = ll.removeprefix("temp ").strip()
                try:
                    T = float(T)
                except ValueError as exc:
                    raise ValueError(
                        "Unable to read temperature from inpsd.dat file."
                    ) from exc

    def run(
        self,
        T: float | u.Quantity | None = None,
        out_dir: pathlib.Path | str = "out",
        **kwargs,
    ) -> None:
        """Run a UppASD calculation.

        If the temperature `T` is not given, the temperature is taken from the input
        file inpsd.dat.

        In particular, a `run_0` directory will be created in the output directory
        `out_dir`. If a directory `run_0` already exists, the next available index will
        be used instead.

        The structure will look like this:

        .. code-block::

            +-- out/
            |   +-- run_0/
            |   +-- run_1/
            |   +-- run_2/

        """
        if T is None:
            T = self.get_T_from_inpsd
        out_dir = pathlib.Path(out_dir)
        run_idx = (
            0
            if not (out_dir / "run_0").is_dir()
            else max(
                [
                    int(i.name.removeprefix("run_"))
                    for i in out_dir.iterdir()
                    if fnmatch.fnmatch(i.name, "run_[0-9]")
                ]
            )
            + 1
        )
        run_dir = pathlib.Path(out_dir) / f"run_{run_idx}"
        run_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy(self.jfile, run_dir)
        shutil.copy(self.momfile, run_dir)
        shutil.copy(self.posfile, run_dir)
        with open(run_dir / "inpsd.dat", "w", encoding="utf-8") as file:
            file.writelines(_parse_inpsd_lines(self.inpsd, temp=str(T), **kwargs))
        t_start = (
            datetime.datetime.now(datetime.UTC)
            .astimezone()
            .isoformat(timespec="seconds")
        )
        _execute_UppASD_in_path(run_dir)
        t_end = (
            datetime.datetime.now(datetime.UTC)
            .astimezone()
            .isoformat(timespec="seconds")
        )
        with open(run_dir / "info.json", "w") as file:
            json.dump(
                {
                    "runner": "run",
                    "temperature": T,
                    "time_start": t_start,
                    "time_end": t_end,
                    "mammos_spindynamics_version": mammos_spindynamics.__version__,
                },
                file,
            )

    def run_temparray(
        self,
        T_array: list | np.ndarray,
        out_dir: pathlib.Path | str = "out",
        **kwargs,
    ):
        """Run a series of UppASD calculations with a temperature array.

        In particular, a `run_temparray_0` directory will be created in the output
        directory `out_dir`. If a directory `run_temparray_0` already exists, the next
        available index will be used instead.

        The structure will look like this:

        .. code-block::

            +-- out/
            |   +-- run_temparray_0/
            |   |   +-- run_0/
            |   |   +-- run_1/
            |   |   +-- run_2/
            |   +-- run_temparray_1/
            |   |   +-- run_0/
            |   |   +-- run_1/

        """
        out_dir = pathlib.Path(out_dir)
        run_idx = (
            0
            if not (out_dir / "run_temparray_0").is_dir()
            else max(
                [
                    int(i.name.removeprefix("run_temparray_"))
                    for i in out_dir.iterdir()
                    if fnmatch.fnmatch(i.name, "run_temparray_[0-9]")
                ]
            )
            + 1
        )
        run_temparray_dir = pathlib.Path(out_dir) / f"run_temparray_{run_idx}"
        run_temparray_dir.mkdir(exist_ok=True, parents=True)
        t_start = (
            datetime.datetime.now(datetime.UTC)
            .astimezone()
            .isoformat(timespec="seconds")
        )
        for T in T_array:
            self.run(
                T=T,
                out_dir=run_temparray_dir,
                **kwargs,
            )
        t_end = (
            datetime.datetime.now(datetime.UTC)
            .astimezone()
            .isoformat(timespec="seconds")
        )
        with open(run_temparray_dir / "info.json", "w") as file:
            json.dump(
                {
                    "runner": "run_temparray",
                    "temperature": T_array,
                    "time_start": t_start,
                    "time_end": t_end,
                    "mammos_spindynamics_version": mammos_spindynamics.__version__,
                },
                file,
            )


# def read_result(...):  # ?
# def read_out_dir(...):  # ?
def get_ResultCollection(out_dir: pathlib.Path | str) -> ResultCollection:
    """Read UppASD calculations results directory."""
    return ResultCollection(out_dir)


class ResultCollection:
    """Collection of UppASD Result instances."""

    def __init__(self, out_dir: pathlib.Path):
        """Initialize ResultCollection given the output directory."""
        self.out_dir = pathlib.Path(out_dir)
        runs = []
        for d_ in self.out_dir.iterdir():
            if fnmatch.fnmatch(d_.name, "run_[0-9]*") or fnmatch.fnmatch(
                d_.name, "run_temparray_[0-9]*"
            ):
                with open(d_ / "info.json") as file:
                    info = json.load(file)
                runs.append({"id": d_.name, **info})
        self.runs = runs

    @property
    def dataframe(self) -> pandas.DataFrame:
        """Dataframe containing information about all available UppASD runs."""
        return pd.DataFrame(self.runs)

    def __getitem__(self, idx):
        """Extract i-th run."""
        if self.runs[idx]["runner"] == "run":
            return Result(self.out_dir / self.runs[idx]["id"])
        elif self.runs[idx]["runner"] == "run_temparray":
            return TempArrayResult(self.out_dir / self.runs[idx]["id"])
        else:
            raise RuntimeError("Runner not recognized.")

    def __str__(self):
        """Dunder method for the `print()` function."""
        return repr(self.dataframe)

    def _repr_html_(self):
        return self.dataframe._repr_html_()


class Result:
    """UppASD Result parser class."""

    def __init__(self, run_dir: pathlib.Path):
        """Initialize Result given the run directory."""
        self.run_dir = pathlib.Path(run_dir)
        with open(self.run_dir / "info.json") as file:
            info = json.load(file)
        self.T = info["temperature"]
        df = pd.read_csv(self.last_cumulant, sep=r"\s+")
        self.data = df.iloc[-1]

    @property
    def inpsd(self) -> str:
        """Return path of ``inpsd.dat`` file."""
        return self.run_dir / "inpsd.dat"

    @property
    def jfile(self) -> str:
        """Return path of ``jfile`` file."""
        return self.run_dir / "jfile"

    @property
    def momfile(self) -> str:
        """Return path of ``momfile`` file."""
        return self.run_dir / "momfile"

    @property
    def posfile(self) -> str:
        """Return path of ``posfile`` file."""
        return self.run_dir / "posfile"

    @property
    def last_cumulant(self) -> str:
        """Return last ``cumulant*.out`` file."""
        cumulant_files = [
            f
            for f in self.run_dir.iterdir()
            if fnmatch.fnmatch(f.name, "cumulant*.out")
        ]
        if not cumulant_files:
            raise ValueError(
                f"No cumulant files found in the `run_dir` {self.run_dir}."
            )
        else:
            cumulant_files.sort()
            return cumulant_files[-1]

    @property
    def volume(self) -> me.Entity:
        """Evaluate cell volume."""
        with open(self.inpsd, encoding="utf-8") as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if fnmatch.fnmatch(line, "cell*"):
                    a = np.genfromtxt(StringIO(line[4:])) * u.Angstrom
                    b = np.genfromtxt(StringIO(lines[i + 1])) * u.Angstrom
                    c = np.genfromtxt(StringIO(lines[i + 2])) * u.Angstrom
                    V = me.Entity("CellVolume", np.dot(a, np.cross(b, c)))
                    break
            else:
                raise RuntimeError("Cell volume could not be evaluated.")
        return V

    @property
    def n_magnetic_atoms(self) -> int:
        """Evaluate number of magnetic atoms."""
        with open(self.momfile, encoding="utf-8") as file:
            n = len(file.readlines())
        return n


class TempArrayResult:
    """Class for the result of the temparray runner."""

    def __init__(self, run_dir: pathlib.Path | str):
        """Initialize TempArrayResult given the run directory."""
        self.run_dir = pathlib.Path(run_dir)
        with open(self.run_dir / "info.json") as file:
            info = json.load(file)
        self.T_array = info["temperature"]
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
        """Dataframe containing information of the temparray run."""
        kB = u.constants.k_B.to("mRy/K").value  # Boltzmann constant in [mRy/K]
        list_data = []
        for sub_run in self.sub_runs:
            list_data.append(pd.concat([pd.Series({"T": sub_run.T}), sub_run.data]))
        df = pd.DataFrame(list_data)
        df["C_v[K_B]"] = np.gradient(df["<E>"] / kB, df["T"], axis=0)
        df = df.rename(columns={"<M>": "<M>[μB]", "#Iter": "iter"})
        return df

    def save_output(self, out_dir: pathlib.Path | str | None = None) -> None:
        """Save output files M(T) and output.csv.

        The generated files are `M(T)` and `output.csv`.
        The first one contains all information evaluated from the cumulant files.
        The file `output.csv` is instead generated from :py:mod:`mammos_entity.io` using
        information from `M(T)` and converting all quantities to ontology units.
        """
        if out_dir is None:
            out_dir = self.run_dir
        out_dir = pathlib.Path(out_dir)
        np.savetxt(
            out_dir / "M(T)",
            self.dataframe.to_numpy(),
            fmt=["%04d"] * 2 + ["% .8E"] * (self.dataframe.shape[1] - 2),
            header="T[K] iter"
            + "".join([f"{h:^16}" for h in self.dataframe.columns[2:]]),
            comments="",
        )
        me.io.entities_to_file(
            out_dir / "output.csv",
            "Magnetization and heat capacity from UppASD",
            T=me.Entity("ThermodynamicTemperature", self.dataframe["T"].to_numpy()),
            Ms=me.Ms(
                self.dataframe["<M>[μB]"].to_numpy()
                * self.n_magnetic_atoms
                * u.constants.muB
                / self.volume.q,
                unit="A/m",
            ),
            U_binder=self.dataframe["U_{Binder}"].to_numpy(),
            Cv=me.Entity(
                "IsochoricHeatCapacity",
                self.dataframe["C_v[K_B]"].to_numpy() * u.constants.k_B,
            ),
        )


def _parse_inpsd_lines(inpsd: pathlib.Path | str, **kwargs):
    """Parse lines of inpsd.dat.

    If a couple `key: value` is given as keyword argument, the line beginning with
    `key` in the `inpsd.dat` will be assigned the value `value`.
    """
    with open(inpsd, encoding="utf-8") as file:
        lines = file.readlines()
    new_lines = []
    for ll in lines:
        for key, val in kwargs.items():
            if fnmatch.fnmatch(ll, f"{key} *"):
                new_lines.append(f"{key} {val}\n")
                break
        else:
            # check if there is any unset `TEMP` left.
            if "TEMP" in ll:
                if "TEMP" in kwargs:
                    ll = ll.replace("TEMP", kwargs["TEMP"])
                elif "temp" in kwargs:
                    ll = ll.replace("TEMP", kwargs["temp"])
                else:
                    raise ValueError("Temperature value not given.")
            new_lines.append(ll)
    return new_lines


def _execute_UppASD_in_path(
    path: pathlib.Path | str,
):
    """Run UppASD calculation in given path."""
    if not pathlib.Path(path).is_dir():
        raise ValueError("Given directory does not exist.")

    res = subprocess.run(
        _uppasd_bin,
        cwd=path,
        stderr=subprocess.PIPE,
    )
    return_code = res.returncode

    if return_code:
        raise RuntimeError(
            f"Simulation has failed. Exit with error: \n{res.stderr.decode('utf-8')}"
        )
