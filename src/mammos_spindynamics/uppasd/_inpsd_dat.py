import re
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
            val = _serialise_cell(val)
        elif key == "ncell":
            val = _serialise_ncell(val)
        serialized[key] = str(val)

    return serialized


def _serialise_cell(val: Any) -> str:
    cell = np.asanyarray(val)  # captures all shape mismatches
    if cell.dtype not in [int, float]:
        raise TypeError(
            f"'cell' elements must be of type int or float, not {cell.dtype}"
        )
    if cell.shape != (3, 3):
        raise ValueError(
            f"'cell' must be a 3x3 matrix or a list of 3 vectors;"
            f"got incompatible shape '{cell.shape}'"
        )
    # additional indentation for lines 2 and 3 to account for 'cell  ' in the
    # first line
    return (
        f"{cell[0, 0]} {cell[0, 1]} {cell[0, 2]}\n"
        f"      {cell[1, 0]} {cell[1, 1]} {cell[1, 2]}\n"
        f"      {cell[2, 0]} {cell[2, 1]} {cell[2, 2]}"
    )


def _serialise_ncell(val: Any) -> str:
    if len(val) != 3:
        raise ValueError(f"ncell must be of length 3, not {len(val)}.")
    if any(not isinstance(elem, int) for elem in val):
        raise TypeError("All elements of ncell must be of type int.")
    return " ".join(map(str, val))


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


def preprocess_inpsd_dat(
    inpsd_dat: Path, simulation_parameters: dict[str, Any]
) -> tuple[str, dict[str, Path]]:
    inp_file = inpsd_dat.read_text()
    files_to_copy = {}
    for key, val in simulation_parameters.items():
        pattern = rf"^{key}\s.*$"
        if key == "cell":
            val = _serialise_cell(val)
            pattern = rf"^{key}\s.*\n.*\n$"
        elif key == "ncell":
            val = _serialise_ncell(val)

        inp_file = re.sub(pattern, f"{key} {val!s}", inp_file, flags=re.MULTILINE)

    for file_ in ["exchange", "posfile", "momfile", "restartfile"]:
        if file_ not in simulation_parameters and (
            match_ := re.search(rf"^{file_}\s+([^\s+]+)", inp_file, flags=re.MULTILINE)
        ):
            # add file name from input file to simulation parameters to be able to use
            # the 'external_file' function
            simulation_parameters[file_] = match_.group(1)

        if file_ == "restartfile":
            init_mag = re.search(r"initmag\s+([^\s]+)", inp_file.lower())
            if not init_mag:
                raise RuntimeError("Missing option 'initmag' in inpsd.dat")
            if int(init_mag.group(1)) != 4:
                # restartfile is only used for initmag 4, so we ignore it for any other
                # value of initmag
                continue

        # stick to the convention of calling the exchange file 'jfile'
        copied_file = "jfile" if file_ == "exchange" else file_
        files_to_copy[copied_file] = external_file(simulation_parameters, file_)
        # hard-coded names for auxilary files to ensure that each run directory is
        # fully self-contained
        inp_file = re.sub(
            rf"^#?{file_}\s.*$",
            f"{file_} ./{copied_file!s}",
            inp_file,
            flags=re.MULTILINE,
        )

    return inp_file, files_to_copy
