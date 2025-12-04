"""UppASD Input class."""

from __future__ import annotations

import fnmatch
import pathlib
from collections.abc import Iterable
from io import StringIO
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import field_validator
from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    pass


STRING_PARAMETERS = {
    "simid",
    "exchange",
    "momfile",
    "posfile",
    "posfiletype",
    "anisotropy",
    "restartfile",
    "ip_mode",
    "do_avrg",
    "do_cumu",
    "do_tottraj",
    "do_sc",
    "do_ams",
    "do_magdos",
    "qpoints",
    "do_stiffness",
}
INT_PARAMETERS = {
    "natoms",
    "ntypes",
    "set_landeg",
    "do_ralloy",
    "Mensemble",
    "tseed",
    "maptype",
    "SDEalgh",
    "Initmag",
    "ip_mcanneal",
    "mcNstep",
    "Nstep",
    "cumu_step",
    "cumu_buff",
    "tottraj_step",
    "plotenergy",
    "magdos_freq",
    "magdos_sigma",
    "eta_max",
    "eta_min",
}
FLOAT_PARAMETERS = {
    "alat",
    "damping",
    "temp",
    "timestep",
}
VECT_PARAMETERS = {
    "BC",
    "cell",
    "ncell",
}


@dataclass()
class Input:
    """UppASD Input class.

    Further information about parameters:
    https://uppasd.github.io/UppASD-manual/input/#inpsd-dat-keywords
    https://github.com/UppASD/UppASD/blob/master/source/Input/inputdata.f90
    """

    cell: Iterable[Iterable[float]]
    """The three lattice vectors describing the cell."""
    ncell: Iterable[float]
    """Number of repetitions of the cell in each of the lattice vector directions."""
    BC: Iterable[str]
    """Boundary conditions (P=periodic, 0=free)."""
    exchange: pathlib.Path = pathlib.Path("./jfile")
    """Location of external file for Heisenberg exchange couplings."""
    momfile: pathlib.Path = pathlib.Path("./momfile")
    """Location of external file describing the magnitudes and directions of magnetic
    moments."""
    posfile: pathlib.Path = pathlib.Path("./posfile")
    """Location of external file for the positions of the atoms in one cell, with the
    site number and type of the atom.."""
    simid: str = "_UppASD_"
    """8 character long simulation id. All output files will include the simid as a
    label."""
    natoms: int | None = None
    """Number of atoms in one cell. (Not needed if a posfile is provided)."""
    ntypes: int | None = None
    """Number of types atoms in one cell. (Not needed if a posfile is provided)."""
    posfiletype: str = "C"
    """Flag to change between C=Cartesian or D=direct coordinates in posfile."""
    set_landeg: int = 0
    """Flag for assigning different values of the gyromagnetic factor for the moments.
    Set to 0 by default."""

    @field_validator("exchange", mode="before")
    @classmethod
    def _check_exchange(cls, exchange: Any) -> Any:
        """Check if exchange file exists."""
        exchange = pathlib.Path(exchange)
        if not exchange.is_file():
            raise FileNotFoundError()
        return exchange

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

    def write(out: pathlib.Path | str) -> None:
        """Write inpsd.dat input file."""
        pass


def parse_inpsd_file(inpsd_file: pathlib.Path | str) -> dict:
    """Parse inpsd.dat file."""
    with open(inpsd_file, encoding="utf-8") as file:
        lines = file.readlines()
    parameters = {}
    for i, line in enumerate(lines):
        for par in STRING_PARAMETERS:
            if fnmatch.fnmatch(line, f"{par}*"):
                parameters[par] = line.split()[1]
        for par in INT_PARAMETERS:
            if fnmatch.fnmatch(line, f"{par}*"):
                try:
                    parameters[par] = int(line.split()[1])
                except ValueError:
                    continue
        for par in FLOAT_PARAMETERS:
            if fnmatch.fnmatch(line, f"{par}*"):
                try:
                    parameters[par] = float(line.split()[1])
                except ValueError:
                    continue
        if fnmatch.fnmatch(line, "BC*"):
            parameters["BC"] = line.removeprefix("BC").split()[:3]
        if fnmatch.fnmatch(line, "cell*"):
            a = np.genfromtxt(StringIO(line.removeprefix("cell")))
            b = np.genfromtxt(StringIO(lines[i + 1]))
            c = np.genfromtxt(StringIO(lines[i + 2]))
            parameters["cell"] = np.vstack((a, b, c))
        if fnmatch.fnmatch(line, "ncell*"):
            parameters["ncell"] = np.genfromtxt(StringIO(line.removeprefix("ncell")))[
                :3
            ]
    return parameters


# def _parse_inpsd_lines(inpsd: pathlib.Path | str, **kwargs):
#     """Parse lines of inpsd.dat.
#
#     If a couple `key: value` is given as keyword argument, the line beginning with
#     `key` in the `inpsd.dat` will be assigned the value `value`.
#     """
#     with open(inpsd, encoding="utf-8") as file:
#         lines = file.readlines()
#     new_lines = []
#     for ll in lines:
#         for key, val in kwargs.items():
#             if fnmatch.fnmatch(ll, f"{key} *"):
#                 new_lines.append(f"{key} {val}\n")
#                 break
#         else:
#             # check if there is any unset `TEMP` left.
#             if "TEMP" in ll:
#                 if "TEMP" in kwargs:
#                     ll = ll.replace("TEMP", kwargs["TEMP"])
#                 elif "temp" in kwargs:
#                     ll = ll.replace("TEMP", kwargs["temp"])
#                 else:
#                     raise ValueError("Temperature value not given.")
#             new_lines.append(ll)
#     return new_lines
