"""Functions for interfacing with UppASD."""

from __future__ import annotations

import datetime
import fnmatch
import pathlib
import shutil
import subprocess
from io import StringIO
from typing import TYPE_CHECKING, Any

import json
import mammos_entity as me
import mammos_spindynamics
import mammos_units as u
import numpy as np
import pandas as pd
from pydantic import field_validator
from pydantic.dataclasses import dataclass

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
        with open(self.inpsd, encoding="utf-8") as f:
            lines = f.readlines()
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
        with open(self.momfile, encoding="utf-8") as f:
            n = len(f.readlines())
        return n

    @property
    def get_T_from_inpsd(self) -> float:
        """Read temperature from inpsd.dat file."""
        with open(self.inpsd, encoding="utf-8") as f:
            lines = f.readlines()
        for ll in lines:
            if fnmatch.fnmatch(ll, "temp *"):
                T = ll.lstrip("temp ").strip()
                try:
                    T = float(T)
                except ValueError as exc:
                    raise ValueError(
                        "Unable to read temperature from inpsd.dat file."
                    ) from exc

    def run(
        self,
        T: float | u.Quantity | None = None,
        outdir: pathlib.Path | str = "out",
        **kwargs,
    ) -> None:
        """Run a UppASD calculation.

        If the temperature `T` is not given, the temperature is taken from the input
        file inpsd.dat.

        In particular, a `run_0` directory will be created in the output directory
        `outdir`. If a directory `run_0` already exists, the next available index will
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
        outdir = pathlib.Path(outdir)
        run_idx = 0 if not (outdir / "run_0").is_dir() else max(
            [int(i.name.lstrip("run_")) for i in outdir.iterdir() if fnmatch.fnmatch(i.name, "run_[0-9]")]
        ) + 1
        run_dir = pathlib.Path(outdir) / f"run_{run_idx}"
        run_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy(self.jfile, run_dir)
        shutil.copy(self.momfile, run_dir)
        shutil.copy(self.posfile, run_dir)
        with open(run_dir / "inpsd.dat", "w", encoding="utf-8") as f:
            f.writelines(_parse_inpsd_lines(self.inpsd, temp=str(T), **kwargs))
        t_start = datetime.datetime.now(datetime.UTC).astimezone().isoformat(timespec="seconds")
        _execute_UppASD_in_path(run_dir)
        t_end = datetime.datetime.now(datetime.UTC).astimezone().isoformat(timespec="seconds")
        with open(run_dir / "info.json", "w") as f:
            json.dump(
                {
                    "runner": "run",
                    "temperature": T,
                    "time_start": t_start,
                    "time_end": t_end,
                    "mammos_spindynamics_version": mammos_spindynamics.__version__,
                },
                f,
            )

    def run_temparray(
        self,
        T_array: list | np.ndarray,
        outdir: pathlib.Path | str = "out",
        **kwargs,
    ):
        """Run a series of UppASD calculations with a temperature array.

        In particular, a `run_temparray_0` directory will be created in the output
        directory `outdir`. If a directory `run_temparray_0` already exists, the next
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
        outdir = pathlib.Path(outdir)
        run_idx = 0 if not (outdir / "run_temparray_0").is_dir() else max(
            [int(i.name.lstrip("run_temparray_")) for i in outdir.iterdir() if fnmatch.fnmatch(i.name, "run_temparray_[0-9]")]
        ) + 1
        run_temparray_dir = pathlib.Path(outdir) / f"run_temparray_{run_idx}"
        run_temparray_dir.mkdir(exist_ok=True, parents=True)
        t_start = datetime.datetime.now(datetime.UTC).astimezone().isoformat(timespec="seconds")
        for T in T_array:
            self.run(
                T=T,
                outdir=run_temparray_dir,
                **kwargs,
            )
        t_end = datetime.datetime.now(datetime.UTC).astimezone().isoformat(timespec="seconds")
        with open(run_temparray_dir / "info.json", "w") as f:
            json.dump(
                {
                    "runner": "run_temparray",
                    "temperature": T_array,
                    "time_start": t_start,
                    "time_end": t_end,
                    "mammos_spindynamics_version": mammos_spindynamics.__version__,
                },
                f,
            )


def read_result(outdir: pathlib.Path | str) -> ResultCollection:
    """Read UppASD calculations results directory."""
    return ResultCollection(outdir)
    # outdir = pathlib.Path(outdir)
    # kB = u.constants.k_B.to("mRy/K").value  # Boltzmann constant in [mRy/K]
    # data = []
    # for run_dir in outdir.iterdir():
    #     if "run_" in run_dir.name:
    #         temp = float(run_dir.name.lstrip("run_"))
    #         cumulant_files = [
    #             f
    #             for f in run_dir.iterdir()
    #             if fnmatch.fnmatch(f.name, "cumulant*.out")
    #         ]
    #         if cumulant_files:
    #             cumulant_files.sort()
    #             df = pd.read_csv(cumulant_files[-1], sep=r"\s+")
    #             row = pd.concat([pd.Series({"T": temp}), df.iloc[-1]])
    #             data.append(row)
    # df = pd.DataFrame(data)
    # df["C_v[K_B]"] = np.gradient(df["<E>"] / kB, df["T"], axis=0)
    # df.to_csv(outdir / "M(T)_df", index=False, float_format="%.8E")
    # out = df[["T", "<M>", "U_{Binder}", "C_v[K_B]"]].rename(
    #     columns={"<M>": "<M>[μB]"}
    # )
    # np.savetxt(
    #     outdir / "M(T)",
    #     out.to_numpy(),
    #     fmt=["%04d"] + ["% .8E"] * 3,
    #     header="T[K] " + "".join([f"{h:^16}" for h in out.columns[1:]]),
    #     comments="",
    # )
    # me.io.entities_to_file(
    #     outdir / "output.csv",
    #     "Magnetization and heat capacity from UppASD",
    #     T=me.Entity("ThermodynamicTemperature", out["T"].to_numpy()),
    #     Ms=me.Ms(
    #         out["<M>[μB]"].to_numpy()
    #         * self.n_magnetic_atoms
    #         * u.constants.muB
    #         / self.volume.q,
    #         unit="A/m",
    #     ),
    #     U_binder=out["U_{Binder}"].to_numpy(),
    #     Cv=me.Entity(
    #         "IsochoricHeatCapacity", out["C_v[K_B]"].to_numpy() * u.constants.k_B
    #     ),
    # )


class ResultCollection:
    """Collection of UppASD Result instances."""

    def __init__(self, outdir: pathlib.Path):
        """Initialize ResultCollection instance."""
        self.outdir = pathlib.Path(outdir)
        runs = []
        for d_ in self.outdir.iterdir():
            if fnmatch.fnmatch(d_.name, "run_[0-9]*") or fnmatch.fnmatch(d_.name, "run_temparray_[0-9]*"):
                with open(d_ / "info.json") as f:
                    info = json.load(f)
                runs.append({"id": d_.name, **info})
        self.runs = runs

    @property
    def dataframe(self) -> pandas.DataFrame:
        """Dataframe containing information about all available UppASD runs."""
        return pd.DataFrame(self.runs)

    def __getitem__(self, idx):
        """Extract i-th run."""
        return Result(self.outdir / self.list_runs[idx])

    def __repr__(self):
        return repr(self.dataframe)

    def _repr_html_(self):
        return self.dataframe._repr_html_()

class Result:
    """UppASD Result parser class."""

    rundir: pathlib.Path

    @property
    def inpsd(self) -> str:
        """Return inpsd.dat file."""
        with open(self.rundir / "inpsd.dat") as f:
            out = f.read()
        return out

    def save_output(self, path: pathlib.Path | str | None = None) -> None:
        """Save output files M(T) and output.csv.

        The generated files are `M(T)` and `output.csv`.
        The first has the following structure:

        .. code-block::

            T[K]     <M>[μB]        U_{Binder}       C_v[K_B]
            0001  0.12345678E-90  1.23456789E-01  2.34567890E+12
            0002  0.12345678E-90  1.23456789E-01  2.34567890E+12

        The file `output.csv` is generated from :py:mod:`mammos_entity.io` using
        information from `M(T)` and converting to ontology units.

        """
        path = pathlib.Path(path)
        kB = u.constants.k_B.to("mRy/K").value  # Boltzmann constant in [mRy/K]
        data = []
        for run_dir in outdir.iterdir():
            if "run_" in run_dir.name:
                temp = float(run_dir.name.lstrip("run_"))
                cumulant_files = [
                    f
                    for f in run_dir.iterdir()
                    if fnmatch.fnmatch(f.name, "cumulant*.out")
                ]
                if cumulant_files:
                    cumulant_files.sort()
                    df = pd.read_csv(cumulant_files[-1], sep=r"\s+")
                    row = pd.concat([pd.Series({"T": temp}), df.iloc[-1]])
                    data.append(row)
        df = pd.DataFrame(data)
        df["C_v[K_B]"] = np.gradient(df["<E>"] / kB, df["T"], axis=0)
        df.to_csv(outdir / "M(T)_df", index=False, float_format="%.8E")
        out = df[["T", "<M>", "U_{Binder}", "C_v[K_B]"]].rename(
            columns={"<M>": "<M>[μB]"}
        )
        np.savetxt(
            outdir / "M(T)",
            out.to_numpy(),
            fmt=["%04d"] + ["% .8E"] * 3,
            header="T[K] " + "".join([f"{h:^16}" for h in out.columns[1:]]),
            comments="",
        )
        me.io.entities_to_file(
            outdir / "output.csv",
            "Magnetization and heat capacity from UppASD",
            T=me.Entity("ThermodynamicTemperature", out["T"].to_numpy()),
            Ms=me.Ms(
                out["<M>[μB]"].to_numpy()
                * self.n_magnetic_atoms
                * u.constants.muB
                / self.volume.q,
                unit="A/m",
            ),
            U_binder=out["U_{Binder}"].to_numpy(),
            Cv=me.Entity(
                "IsochoricHeatCapacity", out["C_v[K_B]"].to_numpy() * u.constants.k_B
            ),
        )



def _parse_inpsd_lines(inpsd: pathlib.Path | str, **kwargs):
    """Parse lines of inpsd.dat.

    If a couple `key: value` is given as keyword argument, the line beginning with
    `key` in the `inpsd.dat` will be assigned the value `value`.
    """
    with open(inpsd, encoding="utf-8") as f:
        lines = f.readlines()
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
