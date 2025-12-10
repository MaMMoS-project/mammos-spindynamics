"""UppASD Input class."""

from __future__ import annotations

import pathlib
import re
from io import StringIO

import numpy as np

STRING_PARAMETERS = {
    "simid",
    "exchange",
    "momfile",
    "posfile",
    "restartfile",
}
FLOAT_PARAMETERS = {
    "alat",
    "temp",
}


def parse_inpsd_file(inpsd_file: pathlib.Path | str) -> dict:
    """Parse inpsd.dat file."""
    with open(inpsd_file, encoding="utf-8") as file:
        lines = file.readlines()
    parameters = {}
    for i, line in enumerate(lines):
        for par in STRING_PARAMETERS:
            if re.match(f"{par} .*", line):
                parameters[par] = line.split()[1]
        for par in FLOAT_PARAMETERS:
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
