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
from ._inpsd_dat import INP_FILE_TEMPLATE, create_input_files, preprocess_inpsd_dat

DEFAULT_SIMULATION_PARAMETERS = {
    "ncell": [25, 25, 25],
    "ip_mcnstep": 25_000,
    "mcnstep": 50_000,
}


class Simulation:
    def __init__(self, inpsd_dat=None, **kwargs):
        self._inpsd_dat = inpsd_dat
        self._simulation_parameters_init = kwargs

    @property
    @staticmethod
    def allowed_parameters(self):
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
        return set(
            key
            for key in self.allowed_parameters
            if key not in DEFAULT_SIMULATION_PARAMETERS
        )

    def __repr__(self):
        args = "".join(
            f"    {key}={val!r},\n" for key, val in self._defined_parameters().items()
        )
        return f"{self.__class__.__name__}(\n    inpsd_dat={self._inpsd_dat},\n{args})"

    def _defined_parameters(self, **kwargs):
        if self._inpsd_dat:
            simulation_parameters = copy.copy(self._simulation_parameters_init)
            simulation_parameters.update(kwargs)
        else:
            simulation_parameters = copy.copy(DEFAULT_SIMULATION_PARAMETERS)
            simulation_parameters.update(copy.copy(self._simulation_parameters_init))
            simulation_parameters.update(kwargs)
        return simulation_parameters

    def create_input_files(self, **kwargs) -> tuple[str, dict[str, Path]]:
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

        if self._inpsd_dat:
            return preprocess_inpsd_dat(Path(self._inpsd_dat), simulation_parameters)
        else:
            if missing_parameters := self.required_parameters - set(
                simulation_parameters.keys()
            ):
                raise RuntimeError(
                    f"The following parameters are missing: {missing_parameters}"
                )
            return create_input_files(**simulation_parameters)

    def run(
        self,
        out: str | Path,
        description: str = "",
        uppasd_executable: str | Path = "uppasd",
        verbosity: int = 1,
        **kwargs,
    ) -> _data.RunData:
        """Run a single UppASD simulation.

        TODO details
        """
        # private method to allow for additional arguments when called from
        # temperature_sweep
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
            "parameters": {key: str(value) for key, value in kwargs.items()},
        }
        _write_inputs(run_path, inp_file_content, files_to_copy, metadata)

        start_time = datetime.datetime.now().isoformat(timespec="seconds")
        if verbosity == 1:
            print(f"Running UppASD in {run_path!s} ...", end="")
        _run_simulation(run_path, uppasd_executable)
        end_time = datetime.datetime.now().isoformat(timespec="seconds")

        if verbosity == 1:
            elapsed_time = datetime.datetime.fromisoformat(
                end_time
            ) - datetime.datetime.fromisoformat(start_time)
            print(f" simulation finished, took {elapsed_time!s}")

        _update_metadata_file(run_path, start_time, end_time)
        return _data.RunData(run_path)

    def temperature_sweep(
        self,
        T: collections.abc.Iterable[numbers.Number],
        out: str | Path,
        restart_with_previous: bool = True,
        description: str = "",
        uppasd_executable: str | Path = "uppasd",
        verbosity: int = 1,
        **kwargs,
    ) -> _data.TemperatureSweepData:
        """Run temperature sweep."""
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
                **{key: str(value) for key, value in kwargs.items()},
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
        for p in base.glob("*")
        if re.match(r"^\d+-(run|temperature_sweep)$", p.name)
    ]
    next_index = max(run_indices) + 1 if run_indices else 0

    next_run_path = base / f"{next_index}-{mode}"
    # The next call would fail if the directory exists already. This should never
    # happen. We can rely on it as additional safety-check to not overwrite anything.
    next_run_path.mkdir()
    return next_run_path, next_index


def _write_inputs(
    run_path: Path,
    inp_file_content: str,
    files_to_copy: dict[str, Path],
    metadata: dict[str, Any],
) -> None:
    (run_path / "inpsd.dat").write_text(inp_file_content)
    for name, orig_path in files_to_copy.items():
        shutil.copy(orig_path, run_path / name)
    with open(run_path / "mammos_spindynamics.yaml", "w") as f:
        yaml.dump(metadata, f)


def _find_executable(uppasd_executable: str) -> Path:
    exe = shutil.which(uppasd_executable)
    if not exe:
        raise RuntimeError(
            f"Could not find UppASD executable with name '{uppasd_executable}' in PATH"
        )
    return Path(exe).resolve()


def _run_simulation(run_dir: Path, uppasd_executable: str):
    with (
        open(run_dir / "uppasd_stdout.txt", "w") as stdout,
        open(run_dir / "uppasd_stderr.txt", "w") as stderr,
    ):
        subprocess.check_call(
            uppasd_executable, cwd=run_dir, stdout=stdout, stderr=stderr
        )

    if "ERROR" in (stdout := (run_dir / "uppasd_stdout.txt").read_text()):
        warnings.warn(
            f"UppASD output contains ERROR lines, simulation likely failed:\n{stdout}",
            stacklevel=3,
        )


def _update_metadata_file(run_path: Path, start_time: str, end_time: str):
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

    elapsed_time = datetime.datetime.fromisoformat(
        end_time
    ) - datetime.datetime.fromisoformat(start_time)
    metadata["metadata"].update(
        {
            "time_start": start_time,
            "time_end": end_time,
            "time_elapsed": str(elapsed_time),
            "uppasd_git_revision": uppasd_git_revision,
        }
    )
    with open(run_path / "mammos_spindynamics.yaml", "w") as f:
        yaml.dump(metadata, f)
