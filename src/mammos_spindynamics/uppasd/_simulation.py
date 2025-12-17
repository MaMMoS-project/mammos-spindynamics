import collections
import copy
import datetime
import numbers
import re
import shutil
import string
import subprocess
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import mammos_spindynamics

from . import _data
from ._inpsd import INP_FILE_TEMPLATE, create_input_files, preprocess_inpsd

DEFAULT_SIMULATION_PARAMETERS = {
    "ncell": [25, 25, 25],
    "ip_mcnstep": 25_000,
    "mcnstep": 50_000,
}


class Simulation:
    """Class to perform UppASD simulations.

    Simulations can be performed without or with existing inpsd.dat file:

    - If no inpsd.dat file is passed the simulation object will use an internal
      template to create a new inpsd.dat file. This template requires a number of
      parameters, which have to be passed by the user either when creating the
      `simulation` object, or when calling :py:func:`~Simulation.run` or
      :py:func:`~Simulation.temperature_sweep`. A list of  required parameters can be
      obtained from :py:func:`~Simulation.required_parameters`.
    - If an inpsd.dat file is passed most of its lines can be modified by passing a new
      value for a parameters as keyword argument. It is not possible to overwrite
      ``ip_mcanneal``.
    """

    def __init__(self, inpsd: Path | str | None = None, **kwargs):
        """Create a new simulation object with an inpsd.dat file and/or parameters.

        Args:
            inpsd: Optional path to an existing inpsd.dat file. If not passed a
                default template defined in the `Simulation` class is used.
            kwargs: Parameters to use (or overwrite when passing `inpsd`) in
                inpsd.dat. Parameters can also be passed when calling
                :py:func:`Simulation.run` or :py:func:`Simulation.temperature_sweep`.
        """
        self._inpsd = inpsd
        self._simulation_parameters_init = kwargs

    @property
    @staticmethod
    def allowed_parameters(self):
        """UppASD inpsd parameters that can be passed without a custom inpsd file."""
        return set(
            key
            for _text, key, _format_spec, _conversion in string.Formatter().parse(
                INP_FILE_TEMPLATE
            )
            if key is not None and key not in ["restartfile_line"]
        )

    @property
    @staticmethod
    def required_parameters(self):
        """UppASD inpsd parameters that must be passed without a custom inpsd file."""
        return set(
            key
            for key in self.allowed_parameters
            if key not in DEFAULT_SIMULATION_PARAMETERS
        )

    def __repr__(self):
        args = "".join(
            f"    {key}={val!r},\n" for key, val in self._defined_parameters().items()
        )
        inpsd = f"    inpsd={self._inpsd},\n" if self._inpsd else ""
        return f"{self.__class__.__name__}(\n{inpsd}{args})"

    def _defined_parameters(self, **kwargs) -> dict[str, Any]:
        """UppASD parameters that are currently defined.

        Merges parameters passed at object creation time, parameters passed as kwargs
        to this method and optionally default parameters if no custom inpsd.dat file
        is used.

        A copy of the dictionary is returned, to ensure that neither default parameters
        nor parameters passed at object creation time are modified.
        """
        if self._inpsd:
            simulation_parameters = copy.copy(self._simulation_parameters_init)
            simulation_parameters.update(kwargs)
        else:
            simulation_parameters = copy.copy(DEFAULT_SIMULATION_PARAMETERS)
            simulation_parameters.update(copy.copy(self._simulation_parameters_init))
            simulation_parameters.update(kwargs)
        return simulation_parameters

    def create_input_files(self, **kwargs) -> tuple[str, dict[str, Path]]:
        """Create input files required for UppASD: inpsd.dat and auxilary files.

        This method creates the content for inpsd.dat and collects auxiliary files
        `posfile`, `momfile`, `exchange`, and optionally `restartfile`. Other external
        files are not supported.

        If no custom inpsd.dat file is used, a predefined template will be used.
        The user has to pass all required template arguments either at object creation
        time or when calling this method. A list of required arguments can be obtained
        from :py:func:`Simulation.required_parameters`.

        If a custom inpsd.dat file is used, the user has the option to overwrite lines
        in that input files by passing the respective keys and new values. Not all
        options in inpsd.dat are currently supported, in particular `ip_mcanneal` cannot
        be used.

        The auxiliary files have hard-coded names in inpsd.dat, when passing a custom
        inpsd.dat the corresponding lines will be modified (trailing comments are lost).
        The dict returned as second object maps names used in the inpsd.dat file to
        original file paths.
        """
        simulation_parameters = self._defined_parameters(**kwargs)

        if T := simulation_parameters.pop("T", None):
            # convenience for the user: set both ip_temp and temp to the same value
            if "ip_temp" in simulation_parameters or "temp" in simulation_parameters:
                raise ValueError(
                    "Parameter 'T' cannot be used simultaneously with parameters"
                    " '(temp, ip_temp)'"
                )
            simulation_parameters["ip_temp"] = T
            simulation_parameters["temp"] = T

        if self._inpsd:
            return preprocess_inpsd(Path(self._inpsd), simulation_parameters)
        else:
            if missing_parameters := self.required_parameters - set(
                simulation_parameters.keys()
            ):
                raise RuntimeError(
                    f"The following parameters are missing: {missing_parameters}"
                )
            return create_input_files(simulation_parameters)

    def run(
        self,
        out: str | Path,
        description: str = "",
        uppasd_executable: str | Path = "uppasd",
        verbosity: int = 1,
        **kwargs,
    ) -> _data.RunData:
        """Run a single UppASD simulation.

        This method creates inputs required for UppASD in a new directory, runs UppASD
        in that directory and returns an object that allows accessing the resulting
        data.

        Args:
            out: Base directory in which the output is stored. The method will
                create a subdirectory ``<index>-run`` inside that directory. The value
                of ``<index>`` depends on the existing directory content. For empty
                directories ``<index>=0``. Otherwise, the method will search for all
                existing ``<some-index>-run`` and ``<some-index>-temperature_sweep``
                directories and use the next index.
            description: Human-readable description of this simulation run, stored as
                metadata.
            uppasd_executable: Name or path to UppASD executable. If provided a name
                the method will search for the executable on PATH.
            verbosity: Verbosity of the run:

                - 0: no output
                - 1: summary line with status information (output directory, runtime)
            kwargs: UppASD arguments to use in the inpsd.dat file, including paths to
                files for momfile, posfile, and exchange. Further details are explained
                in :py:func:`Simulation.create_input_files`.

                In addition a special argument ``T`` is supported, which will be used to
                set both ``ip_temp`` and ``temp``. Passing both ``T`` and (at least) one
                of ``ip_temp`` or ``temp`` is not supported.

                Parameters passed to `run` instead of at object creation time are
                recorded as separate metadata and can be used later on to conveniently
                retrieve simulation results.

        Returns:
            An object to access UppASD outputs.
        """
        out = Path(out)
        uppasd_executable = _find_executable(uppasd_executable)

        inp_file_content, files_to_copy = self.create_input_files(**kwargs)

        run_path, index = _create_run_dir(out)

        metadata = {
            "metadata": {
                "mammos_spindynamics_version": mammos_spindynamics.__version__,
                "mode": "run",
                "description": description,
                "index": index,
            },
            "parameters": {key: _sanitize_val(value) for key, value in kwargs.items()},
        }
        _write_inputs(run_path, inp_file_content, files_to_copy, metadata)

        time_start = datetime.datetime.now().isoformat(timespec="seconds")
        if verbosity == 1:
            print(f"Running UppASD in {run_path!s} ...", end=" ")
        _run_simulation(run_path, uppasd_executable)
        time_end = datetime.datetime.now().isoformat(timespec="seconds")
        time_elapsed = datetime.datetime.fromisoformat(
            time_end
        ) - datetime.datetime.fromisoformat(time_start)

        if verbosity == 1:
            print(f"simulation finished, took {time_elapsed!s}")

        _update_metadata_file(run_path, time_start, time_end, time_elapsed)
        return _data.RunData(run_path)

    def temperature_sweep(
        self,
        T: collections.abc.Iterable[numbers.Real],
        out: str | Path,
        restart_with_previous: bool = True,
        description: str = "",
        uppasd_executable: str | Path = "uppasd",
        verbosity: int = 2,
        **kwargs,
    ) -> _data.TemperatureSweepData:
        """Run temperature sweep.

        This method runs consecutive simulations for multiple temperatures. For each
        temperature it will call :py:func:`Simulation.run` internally. The method does
        not sort the temperatures, so make sure you pass the desired order, in
        particular when using `restart_with_previous=True`.

        After finishing all simulations two additional files `M(T)` and `output.csv`
        with aggregated simulation results are created.

        Args:
            T: Temperatures in Kelvin to run the simulation for. The value will be used
                for both ``ip_temp`` and ``temp``.
            out: Base directory for the output. A new directory
                ``<index>-temperature_sweep`` will be created inside, the logic is
                equivalent to :py:func:`Simulation.run`.
            restart_with_previous: If set to True use the restart file from the previous
                run as initial configuration for all but the first run in the sweep.
            description: Human-readable description of the sweep, the individual runs
                do not have a description.
            uppasd_executable: Name or path to UppASD executable, like in
                :py:func:`Simulation.run`.
            verbosity: Verbosity of the sweep:

                - 0: no output
                - 1: only summary output for the whole sweep (number and list of
                  temperatures)
                - 2: summary for the sweep and summary for each run (verbosity 1 for
                  each run)
            kwargs: UppASD arguments in inpsd.dat as in :py:func:`Simulation.run`. The
                two options ``ip_temp`` and ``temp`` cannot be used as instead ``T`` is
                required.

        Returns:
            An object to access UppASD outputs.

        """
        run_path, index = _create_run_dir(Path(out), mode="temperature_sweep")

        # convert any form of T to a list of T values
        Ts = np.asanyarray(T).tolist()

        metadata = {
            "metadata": {
                "description": description,
                "index": index,
                "mode": "temperature_sweep",
            },
            "parameters": {
                "T": Ts,
                **{key: _sanitize_val(value) for key, value in kwargs.items()},
            },
        }
        with open(run_path / "mammos_spindynamics.yaml", "w") as f:
            yaml.dump(metadata, f)

        if verbosity >= 1:
            print(
                f"Running simulations for {len(Ts)} different temperatures:\n    {Ts!s}"
            )

        # run first simulation with default options; later simulations optionally with
        # restarting from previous
        if verbosity >= 2:
            print(f"T={Ts[0]}:", end=" ")
        run_data = self.run(
            T=Ts[0],
            out=run_path,
            uppasd_executable=uppasd_executable,
            verbosity=verbosity - 1,
            **kwargs,
        )
        for T_ in Ts[1:]:
            if restart_with_previous:
                kwargs.update({"initmag": 4, "restartfile": run_data.restartfile})
            if verbosity >= 2:
                print(f"T={T_}:", end=" ")
            run_data = self.run(
                T=T_,
                out=run_path,
                uppasd_executable=uppasd_executable,
                verbosity=verbosity - 1,
                **kwargs,
            )

        result = _data.TemperatureSweepData(run_path)
        result.save_output(run_path)
        return result


def _create_run_dir(base: Path, mode="run") -> tuple[Path, int]:
    """Create a new directory for run or temperature_sweep.

    The directory name is prefixed with an index, which enumerates all runs and
    temperature_sweeps. The function looks for the maximum existing index and uses
    the next index.
    """
    if mode not in ["run", "temperature_sweep"]:
        raise ValueError(
            f"Mode {mode} not supported, must be 'run' or 'temperature_sweep'"
        )

    if not base.exists():
        base.mkdir(parents=True)
    elif base.is_file():
        raise RuntimeError(f"The path '{base}' passed as output directory is a file.")

    if not (base / "mammos_spindynamics.yaml").exists():
        with open(base / "mammos_spindynamics.yaml", "w") as f:
            yaml.dump({"metadata": {"mode": "mammos_uppasd_data"}}, f)

    run_indices = [
        int(p.name.split("-")[0])
        for p in base.iterdir()
        if re.match(r"^\d+-(run|temperature_sweep)$", p.name)
    ]
    next_index = max(run_indices) + 1 if run_indices else 0

    next_run_path = base / f"{next_index}-{mode}"
    # The next call would fail if the directory exists already. This should never
    # happen. We can rely on it as additional safety-check to not overwrite anything.
    next_run_path.mkdir()
    return next_run_path, next_index


def _write_inputs(
    out_path: Path,
    inp_file_content: str,
    files_to_copy: dict[str, Path],
    metadata: dict[str, Any],
) -> None:
    """Write input files.

    The function writes the file inpsd.dat, copies all files in ``files_to_copy``,
    where keys are the new filenames and values are paths to the original files, and
    create a ``mammos_spindynamics.yaml`` file with ``metadata``.

    The directory ``out_path`` must already exist.
    """
    (out_path / "inpsd.dat").write_text(inp_file_content)
    for name, orig_path in files_to_copy.items():
        shutil.copy(orig_path, out_path / name)
    with open(out_path / "mammos_spindynamics.yaml", "w") as f:
        yaml.dump(metadata, f)


def _find_executable(uppasd_executable: str) -> Path:
    """Find executable with given name in PATH."""
    exe = shutil.which(uppasd_executable)
    if not exe:
        raise RuntimeError(
            f"Could not find UppASD executable with name '{uppasd_executable}' in PATH"
        )
    return Path(exe).resolve()


def _run_simulation(run_dir: Path, uppasd_executable: str):
    """Call UppASD in ``run_dir``.

    Stdout and stderr are redirected to files ``uppasd_stdout.txt`` and
    ``uppasd_stderr.txt``. The method will raise an exception if UppASD fails and print
    a warning if stdout contains the word ERROR, which is often the case when running
    a simulation with an invalid input file (UppASD does typically not fail in such
    cases).
    """
    res = subprocess.run(
        uppasd_executable,
        cwd=run_dir,
        check=True,
        capture_output=True,
        text=True,
    )

    if res.stderr:
        warnings.warn(
            f"Stderr is not empty, the simulation may have failed:\n{res.stderr}",
            stacklevel=3,
        )

    if "ERROR" in res.stdout:
        warnings.warn(
            "UppASD output contains ERROR lines, simulation likely failed:\n"
            + res.stdout,
            stacklevel=3,
        )

    (run_dir / "uppasd_stdout.txt").write_text(res.stdout)
    (run_dir / "uppasd_stderr.txt").write_text(res.stderr)


def _update_metadata_file(
    run_path: Path, time_start: str, time_end: str, time_elapsed: str
) -> None:
    """Update ``mammos_spindynamics.yaml`` metadata file.

    Inserts:
      - start/end/elapsed time
      - if available the git revision reported by UppASD
    """
    # convert to string first to limit resolution to seconds
    with open(run_path / "mammos_spindynamics.yaml") as f:
        metadata = yaml.safe_load(f)

    uppasd_yaml = list(run_path.glob("uppasd.*.yaml"))
    uppasd_yaml = None
    if uppasd_yaml:
        with open(uppasd_yaml[0]) as f:
            uppasd_git_revision = yaml.safe_load(f)["git_revision"]
    else:
        uppasd_git_revision = "<unknown>"

    metadata["metadata"].update(
        {
            "time_start": time_start,
            "time_end": time_end,
            "time_elapsed": str(time_elapsed),
            "uppasd_git_revision": uppasd_git_revision,
        }
    )
    with open(run_path / "mammos_spindynamics.yaml", "w") as f:
        yaml.dump(metadata, f)


def _sanitize_val(val: Any) -> Any:
    """Convert some types to improve yaml dump.

    Converts:
    - tuple -> list
    - Path -> str
    """
    # TODO replace with custom YamlDumper
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, tuple):
        return list(val)
    return val
