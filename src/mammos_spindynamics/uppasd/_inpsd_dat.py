from pathlib import Path
from typing import Any

import numpy as np

INP_FILE_TEMPLATE = """\
cell  {cell}
alat  {alat}
ncell {ncell}
bc    P P P
sym   0

posfile     ./posfile
posfiletype {posfiletype}
momfile     ./momfile
maptype     {maptype}
exchange    ./jfile

initmag {initmag}
{restartfile_line}

ip_mode    M
ip_temp    {ip_temp}
ip_mcnstep {ip_mcnstep}

mode    M
temp    {temp}
mcnstep {mcnstep}

plotenergy   1
do_proj_avrg Y
do_cumu      Y
"""


def serialize_parameters(parameters: dict[str, Any]) -> dict[str, str]:
    """Serialize parameters for input file and validate selected.

    Most parameters are simply converted to string by calling `str(value)`. No further
    checks are performed for these.

    The following parameters are treated special:
    - cell: check that it is a 3x3 matrix of numbers and convert to a multi-line string
    - ncell: check that it is a 3 vector of int and convert to a list in uppasd format
          (elements separated by spaces, no brackets)
    """
    serialized = {}
    for key, val in parameters.items():
        if key == "cell":
            cell = np.asanyarray(val)  # captures all shape mismatches
            if cell.shape != (3, 3):
                raise ValueError(
                    f"'cell' must be a 3x3 matrix or a list of 3 vectors;"
                    f"got incompatible shape '{cell.shape}'"
                )
            if cell.dtype not in [int, float]:
                raise TypeError(
                    f"'cell' elements must be of type int or float, not {cell.dtype}"
                )
            # additional indentation for lines 2 and 3 to account for 'cell  ' in the
            # first line
            val = (
                f"{cell[0, 0]} {cell[0, 1]} {cell[0, 2]}\n"
                f"      {cell[1, 0]} {cell[1, 1]} {cell[1, 2]}\n"
                f"      {cell[2, 0]} {cell[2, 1]} {cell[2, 2]}"
            )
        elif key == "ncell":
            assert len(val) == 3 and all(isinstance(elem, int) for elem in val)
            val = " ".join(map(str, val))
        serialized[key] = str(val)

    return serialized


def create_input_files(out: Path, **kwargs) -> tuple[str, dict[str, Path]]:
    simulation_parameters = kwargs

    # hard-coded names for exchange, posfile, momfile
    files_to_copy = {}
    for file_ in ["exchange", "posfile", "momfile"]:
        files_to_copy[file_] = external_file(simulation_parameters, file_)

    if simulation_parameters["initmag"] == 4:
        file_ = "restartfile"
        if file_ not in simulation_parameters:
            raise AttributeError("restartfile for initmag 4 missing")
        files_to_copy[file_] = external_file(simulation_parameters, file_)
        # restartfile needs to be passed as a whole line to the template
        simulation_parameters["restartfile_line"] = "restartfile ./restartfile"
    else:
        simulation_parameters["restartfile_line"] = ""
    inp_file = INP_FILE_TEMPLATE.format(**serialize_parameters(simulation_parameters))

    return inp_file, files_to_copy


def external_file(simulation_parameters: dict[str, Any], key: str) -> Path:
    """Find file pased for UppASD argument key.

    The key is removed from simulation_parameters, because all files have hard-coded
    relative paths and names in the input file template.
    """
    try:
        file_path = simulation_parameters.pop(key)
    except KeyError:
        raise AttributeError(f"Missing parameter: file {key} not passed") from None
    if not isinstance(key, str | Path):
        raise ValueError(
            f"Invalid type for {key}; must be str or Path, not {type(file_path)}."
        )
    file_path = Path(file_path)
    if not file_path.is_file():
        raise ValueError(f"File '{file_path!s}' passed for {key} does not exist")

    return file_path
