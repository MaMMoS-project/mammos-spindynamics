import re
import warnings
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
exchange    ./exchange
maptype     {maptype}

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
    - ncell: check that it is a length 3 array_like and convert to a strin in uppasd
          format (elements separated by spaces, no brackets)
    """
    serialized = {}
    for key, val in parameters.items():
        if key == "cell":
            val = _serialise_cell(val)
        elif key == "ncell":
            val = _serialize_array("ncell", val)
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
            f"'cell' must be a 3x3 matrix or a list of 3 vectors; "
            f"got incompatible shape '{cell.shape}'"
        )
    # additional indentation for lines 2 and 3 to account for 'cell  ' in the
    # first line
    return (
        f"{cell[0, 0]} {cell[0, 1]} {cell[0, 2]}\n"
        f"      {cell[1, 0]} {cell[1, 1]} {cell[1, 2]}\n"
        f"      {cell[2, 0]} {cell[2, 1]} {cell[2, 2]}"
    )


def _serialize_array(name: str, val: Any, length: int = 3) -> str:
    if len(val) != length:
        raise ValueError(f"'{name}' must be of length {length}, not {len(val)}.")
    return " ".join(map(str, val))


def create_input_files(simulation_parameters) -> tuple[str, dict[str, Path]]:
    """Create inpsd.dat content and collect auxiliary files.

    Returns:
        - The content of inpsd.dat as a string.
        - A dictionary of auxilary files; keys are the names used in the run directory
            and the inpsd.dat file, values are `Path`s to the original files.
    """
    # first process all files; the `external_file` function modifies kwargs
    # to remove user-given paths to files as they have hard-coded names in the
    # inpsd.dat template

    # hard-coded names for exchange, posfile, momfile
    # these three are always required, other files are not supported in the Python
    # interface
    files_to_copy = {}
    for file_ in ["exchange", "posfile", "momfile"]:
        files_to_copy[file_] = _external_file(simulation_parameters, file_)

    if simulation_parameters.get("initmag") == 4:
        file_ = "restartfile"
        if file_ not in simulation_parameters:
            raise AttributeError("restartfile for initmag 4 missing")
        files_to_copy[file_] = _external_file(simulation_parameters, file_)
        # restartfile information needs to be passed as a whole line to the template
        simulation_parameters["restartfile_line"] = "restartfile ./restartfile"
    else:
        simulation_parameters["restartfile_line"] = ""

    inp_file = INP_FILE_TEMPLATE.format(**serialize_parameters(simulation_parameters))

    return inp_file, files_to_copy


def _external_file(simulation_parameters: dict[str, Any], key: str) -> Path:
    """Find file passed for UppASD argument key.

    The key is removed from simulation_parameters, because all files have hard-coded
    relative paths and names in the input file template.
    """
    try:
        file_path = simulation_parameters.pop(key)
    except KeyError:
        raise AttributeError(f"Missing argument '{key}'.") from None
    if not isinstance(file_path, str | Path):
        raise ValueError(
            f"Invalid type for '{key}': must be str or Path, not {type(file_path)}."
        )
    file_path = Path(file_path)
    if not file_path.is_file():
        raise ValueError(f"File '{file_path!s}' passed for '{key}' does not exist.")

    return file_path


def preprocess_inpsd(
    inpsd: Path, simulation_parameters: dict[str, Any]
) -> tuple[str, dict[str, Path]]:
    """Preprocess an existing inpsd.dat file.

    Users can overwrite lines in the input file by passing the respective key and a
    new value. Comments in the old lines will be lost.

    Names for posfile, momfile and exchange are always replaced with the defaults.
    If they are not in simulation_parameters, their values are read from the given
    inpsd file.

    Most arguments allowed in the inpsd.dat template are supported. In particular cell,
    ncell and bc are handled correctly, BUT they must be given in lower case.

    Currently unsupported are multi-line options such as ip_mcanneal. Passing these as
    in simulation_parameters will result in syntactically wrong input files without
    any notice. Running a simulation with such a file should typically cause some sort
    of failure of UppASD, so the user should notice.

    Returns:
        - The content of inpsd.dat as a string.
        - A dictionary of auxilary files; keys are the names used in the run directory
            and the inpsd.dat file, values are `Path`s to the original files.
    """
    inp_file = inpsd.read_text()
    files_to_copy = {}
    for key, val in simulation_parameters.items():
        pattern = rf"^{key}\s.*$"
        if key == "cell":
            val = _serialise_cell(val) + "\n"  # TODO is this newline required?
            # replace three consecutive lines, where the first line starts with 'cell '
            pattern = rf"^{key}\s.*\n.*\n.*$"  # TODO write a test for this
        elif key in ["ncell", "bc"]:
            val = _serialize_array(key, val)

        val = str(val)

        if "\\" in val:
            # paths on Windows contain backslashes
            val = re.escape(val)
        inp_file = re.sub(pattern, rf"{key} {val}", inp_file, flags=re.MULTILINE)

    # only the following four files are supported, restartfile is only dealt with
    # properly when initmag is set to 4
    for file_ in ["exchange", "posfile", "momfile", "restartfile"]:
        if file_ not in simulation_parameters and (
            # find file name and path in input file, ignore trailing comment (not sure
            # if UppASD allows one in file lines)
            match_ := re.search(rf"^{file_}\s+([^\s+]+)", inp_file, flags=re.MULTILINE)
        ):
            # add file name from input file to simulation parameters to be able to use
            # the 'external_file' function
            simulation_parameters[file_] = match_.group(1)

        if file_ == "restartfile":
            init_mag = re.search(r"initmag\s+([^\s]+)", inp_file.lower())
            if not init_mag:
                raise RuntimeError("Missing option 'initmag' in inpsd.dat.")
            if int(init_mag.group(1)) != 4:
                # restartfile is only used for initmag 4, so we ignore it for any other
                # value of initmag
                if file_ in simulation_parameters or (match_ and match_.group(1)):
                    warnings.warn(
                        "The parameter 'restartfile' is ignored for initmag!=4, got"
                        f" initmag {init_mag.group(1)}.",
                        stacklevel=3,
                    )
                continue

        files_to_copy[file_] = _external_file(simulation_parameters, file_)
        # insert hard-coded names for auxilary files to ensure that each run directory
        # is fully self-contained
        inp_file = re.sub(
            rf"^{file_}\s.*$",
            f"{file_} ./{file_!s}",
            inp_file,
            flags=re.MULTILINE,
        )

    return inp_file, files_to_copy
