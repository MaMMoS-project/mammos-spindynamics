"""UppASD Simulation class."""

from __future__ import annotations

import datetime
import fnmatch
import yaml
import pathlib
import shutil
import subprocess
from io import StringIO
from typing import TYPE_CHECKING, Any, Iterable

import mammos_entity as me
import mammos_units as u
import numpy as np
import pandas as pd
from pydantic import field_validator
from pydantic.dataclasses import dataclass

import mammos_spindynamics
from mammos_spindynamics.uppasd.inpsd import Input, parse_inpsd_file
from mammos_spindynamics.uppasd.data import RunData, TemperatureSweepData

if TYPE_CHECKING:
    import mammos_units
    import numpy
    import pandas


_uppasd_bin = shutil.which("uppasd")
if _uppasd_bin is None:
    raise RuntimeError(
        "uppasd not found. "
        "Please install it from conda-forge using `conda install conda-forge::uppasd `"
        "If uppasd was compiled from source, please create a soft link, e.g. by using "
        "`ln -s <binary_location> ~/.local/bin/uppasd`."
    )


@dataclass()
class Simulation:
    """UppASD Simulation class."""

    def __init__(self, **kwargs):
        self.input = Input(**kwargs)

    @classmethod
    def from_inpsd(cls, inpsd_file, **kwargs):
        inpsd_dict = parse_inpsd_file(inpsd_file)
        sim_dict = {**inpsd_dict, **kwargs}
        return cls(**sim_dict)

    def run(
        self,
        T: mammos_units.Quantity | float,
        out: pathlib.Path | str = "out",
        **kwargs,
    ) -> None:
        """Run a UppASD calculation.

        In particular, a `run-0` directory will be created in the output directory
        `out`. If a directory `run-0` already exists, the next available index will
        be used instead.

        The structure will look like this:

        .. code-block::

            +-- out/
            |   +-- run-0/
            |   |   +-- jfile
            |   |   +-- posfile
            |   |   +-- momfile
            |   |   +-- inpsd.dat
            |   |   +-- cumulants._UppASD_.out
            |   |   +-- ...
            |   |   +-- mammos_spindynamics_info.{yaml, toml}
            |   +-- run-1/
            |   +-- run-2/

        """
        run_dir = _get_available_out_dir(pathlib.Path(out), "run")
        shutil.copy(self.jfile, run_dir)
        shutil.copy(self.momfile, run_dir)
        shutil.copy(self.posfile, run_dir)
        self.input.T = T
        parameters = {"T": T, **kwargs}
        for key, value in kwargs:
            setattr(self.input, key, value)
        self.input.write(run_dir / "inpsd.dat")

        time_start = datetime.datetime.now(datetime.UTC).astimezone()
        _execute_UppASD_in_path(run_dir)
        time_end = datetime.datetime.now(datetime.UTC).astimezone()
        time_elapsed = time_end - time_start
        with open(run_dir / "info.yaml", "w") as file:
            yaml.dump(
                {
                    "metadata": {
                        "UppASD_version": "v6.0.2",  # Not parsable from output files?
                        "mammos_spindynamics_version": mammos_spindynamics.__version__,
                        "mode": "run",
                        "time_start": time_start.isoformat(timespec="seconds"),
                        "time_end": time_end.isoformat(timespec="seconds"),
                        "time_elapsed": time_elapsed,
                    },
                    "parameters": parameters
                },
                file,
            )

        return RunData(run_dir)

    def temperature_sweep(
        self,
        T: Iterable[mammos_units.Quantity | float],
        restart_with_previous: bool = False,
        out: pathlib.Path | str = "out",
        comment: str = "",
        **kwargs,
    ):
        """Run a series of UppASD calculations with a list of temperatures.

        In particular, a `temperature_sweep-0` directory will be created in the output
        directory `out`. If a directory with such name already exists, the next
        available index will be used instead.

        The structure will look like this:

        .. code-block::

            +-- out/
            |   +-- temperature_sweep-0/
            |   |   +-- run-0/
            |   |   |   +-- jfile
            |   |   |   +-- posfile
            |   |   |   +-- momfile
            |   |   |   +-- inpsd.dat
            |   |   |   +-- cumulants._UppASD_.out
            |   |   |   +-- ...
            |   |   |   +-- mammos_spindynamics_info.{yaml, toml}
            |   |   +-- run-1/
            |   |   +-- run-2/
            |   |   +-- mamos_spindynamics_info.{yaml,toml}
            |   |   +-- M(T)
            |   +-- temperature_sweep-1/
            |   |   +-- run-0/
            |   |   +-- run-1/

        """
        sweep_dir = _get_available_out_dir(pathlib.Path(out), "temperature_sweep")
        time_start = (
            datetime.datetime.now(datetime.UTC)
            .astimezone()
            .isoformat(timespec="seconds")
        )
        for T_i in T:
            self.run(
                T=T_i,
                out=sweep_dir,
                **kwargs,
            )
        time_end = (
            datetime.datetime.now(datetime.UTC)
            .astimezone()
            .isoformat(timespec="seconds")
        )
        with open(sweep_dir / "info.yaml", "w") as file:
            yaml.dump(
                {
                    "UppASD_version": "v6.0.2",  # Not parsable from output files?
                    "mammos_spindynamics_version": mammos_spindynamics.__version__,
                    "mode": "temperature_sweep",
                    "temperature": T,
                    "time_start": time_start,
                    "time_end": time_end,
                },
                file,
            )

        return TemperatureSweepData(sweep_dir)


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


def _get_available_out_dir(out, prefix):
    """Get an available output path.

    Returns `out/prefix-0` if such directory does not exist. Otherwise, if `idx` is the
    highest integer such that `out/prefix-idx` exists, this function returns the path
    `out/prefix-(idx+1)`.
    """
    if not (out / f"{prefix}-0").is_dir():
        idx = 0
    else:
        indices = [
            int(directory.name.removeprefix(f"{prefix}-"))
            for directory in out.iterdir()
            if fnmatch.fnmatch(directory.name, f"{prefix}-[0-9]")
        ]
        idx = max(indices) + 1
    out_dir = out / f"{prefix}-{idx}"
    out_dir.mkdir(exist_ok=True, parents=True)
    return out_dir


def _get_formatted_time():
    """Get time formatted in UTC and seconds."""
    return datetime.datetime.now(datetime.UTC).astimezone().isoformat(timespec="seconds")
