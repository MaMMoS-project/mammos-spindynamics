"""Functions for interfacing with UppASD."""

import fnmatch
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

if TYPE_CHECKING:
    pass

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
    def T_from_inpsd(self) -> float:
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
        T_array: list | np.ndarray | None = None,
        outdir: pathlib.Path | str = "UppASD",
        **kwargs,
    ) -> None:
        """Run a series of UppASD calculation on different temperatures.

        If the temperature array `temp_array` is not given, the temperature is taken
        from the input file inpsd.dat.

        In particular, a result directory will be created for each temperature following
        the structure:

        .. code-block::

            +-- outdir/
            |   +-- run_1/
            |   +-- run_10/
            |   +-- run_20/

        """
        outdir = pathlib.Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        if T_array is None:
            T_array = [self.T_from_inpsd]
        for T in T_array:
            outdir_T = pathlib.Path(outdir) / f"run_{T}"
            outdir_T.mkdir(exist_ok=True, parents=True)
            shutil.copy(self.jfile, outdir_T)
            shutil.copy(self.momfile, outdir_T)
            shutil.copy(self.posfile, outdir_T)
            with open(outdir_T / "inpsd.dat", "w", encoding="utf-8") as f:
                f.writelines(_parse_inpsd_lines(self.inpsd, temp=T, **kwargs))
            _execute_UppASD_in_path(outdir_T)
        self.parse_outdir(outdir)

    def parse_outdir(self, outdir: pathlib.Path | str) -> None:
        r"""Create `M(T)` file from different calculations.

        In particular, we expect the directory `outdir` to have the following structure:

        .. code-block::

            +-- outdir/
            |   +-- run_1/
            |   +-- run_10/
            |   +-- run_20/

        The generated files are `M(T)` and `output.csv`.
        The first has the following structure:

        .. code-block::

            T[K]     <M>[μB]        U_{Binder}       C_v[K_B]
            0001  0.12345678E-90  1.23456789E-01  2.34567890E+12
            0002  0.12345678E-90  1.23456789E-01  2.34567890E+12

        The file `output.csv` is generated from :py:mod:`mammos_entity.io` using
        information from `M(T)` and converting to ontology units.
        """
        outdir = pathlib.Path(outdir)
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
