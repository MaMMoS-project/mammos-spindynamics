"""UppASD Simulation class."""

from __future__ import annotations

import datetime
import pathlib
import re
import shutil
import subprocess
from collections.abc import Iterable
from typing import TYPE_CHECKING

import yaml

import mammos_spindynamics
from mammos_spindynamics.uppasd.data import RunData, TemperatureSweepData
from mammos_spindynamics.uppasd.inpsd import Input, parse_inpsd_file

if TYPE_CHECKING:
    import mammos_units


_uppasd_bin = shutil.which("uppasd")
if _uppasd_bin is None:
    raise RuntimeError(
        "uppasd not found. "
        "Please install it from conda-forge using `conda install conda-forge::uppasd `"
        "If uppasd was compiled from source, please create a soft link, e.g. by using "
        "`ln -s <binary_location> ~/.local/bin/uppasd`."
    )


class Simulation:
    """UppASD Simulation class."""

    def __init__(self, **kwargs):
        """Initialize simulation instance."""
        self.simulation_parameters = kwargs
        # self.input = Input(**self.simulation_parameters)

    def __repr__(self):
        """Define repr."""
        return f"Simulation({repr(self.simulation_parameters)})"

    @classmethod
    def from_inpsd(cls, inpsd_file, **kwargs):
        """Define Simulation instance from a given `inpsd.dat` file."""
        inpsd_dict = parse_inpsd_file(inpsd_file)
        sim_dict = {**inpsd_dict, **kwargs}
        return cls(**sim_dict)

    def run(
        self,
        T: mammos_units.Quantity | float,
        out: pathlib.Path | str = "out",
        description: str = "",
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
        # Define output directory
        out = pathlib.Path(out)
        dir_idx = _get_available_out_dir(pathlib.Path(out), "run")
        run_dir = out / f"run-{dir_idx}"
        run_dir.mkdir(parents=True)

        # Set run parameters and define input object
        run_parameters = {"T": T, **kwargs}
        input = Input(
            **self.simulation_parameters,
            temp=T,
            **kwargs,
        )

        # Write inpsd.dat input file
        input.write(run_dir / "inpsd.dat")

        # Execute UppASD and record time
        time_start = datetime.datetime.now(datetime.UTC).astimezone()
        _execute_UppASD_in_path(run_dir)
        time_end = datetime.datetime.now(datetime.UTC).astimezone()
        time_elapsed = time_end - time_start

        # Create information yaml
        with open(run_dir / "info.yaml", "w") as file:
            yaml.dump(
                {
                    "metadata": {
                        "index": dir_idx,
                        "description": description,
                        "UppASD_version": "v6.0.2",  # Not parsable from output files?
                        "mammos_spindynamics_version": mammos_spindynamics.__version__,
                        "mode": "run",
                        "time_start": time_start.isoformat(timespec="seconds"),
                        "time_end": time_end.isoformat(timespec="seconds"),
                        "time_elapsed": str(time_elapsed),
                    },
                    "parameters": run_parameters,
                },
                file,
            )

        # Update mammos_uppasd_data yaml
        _update_mammos_uppasd_data(out, run_dir.name)

        return RunData(run_dir)

    def temperature_sweep(
        self,
        T: Iterable[mammos_units.Quantity | float],
        restart_with_previous: bool = False,
        out: pathlib.Path | str = "out",
        description: str = "",
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
        # Define output directory
        out = pathlib.Path(out)
        dir_idx = _get_available_out_dir(pathlib.Path(out), "temperature_sweep")
        sweep_dir = out / f"temperature_sweep-{dir_idx}"
        sweep_dir.mkdir(parents=True)

        # Set sweep parameters
        parameters = {"T": list(T), **kwargs}

        # Execute temperature sweep and record time
        run_i = None
        time_start = datetime.datetime.now(datetime.UTC).astimezone()
        for i, T_i in enumerate(T):
            if restart_with_previous and i > 0:
                kwargs["restartfile"] = str(run_i.restart_file)
            run_i = self.run(
                T=T_i,
                out=sweep_dir,
                **kwargs,
            )
        time_end = datetime.datetime.now(datetime.UTC).astimezone()
        time_elapsed = time_end - time_start

        # Create sweep yaml
        with open(sweep_dir / "info.yaml", "w") as file:
            yaml.dump(
                {
                    "metadata": {
                        "index": dir_idx,
                        "description": description,
                        "UppASD_version": "v6.0.2",  # Not parsable from output files?
                        "mammos_spindynamics_version": mammos_spindynamics.__version__,
                        "mode": "temperature_sweep",
                        "temperature": list(T),
                        "time_start": time_start.isoformat(timespec="seconds"),
                        "time_end": time_end.isoformat(timespec="seconds"),
                        "time_elapsed": str(time_elapsed),
                    },
                    "parameters": parameters,
                },
                file,
            )

        # Update mammos_uppasd_data yaml
        _update_mammos_uppasd_data(out, sweep_dir.name)

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
    """Get the index for an available output path.

    Returns `0` if the directory `out/<prefix>-0` does not exist. Otherwise, returns
    `idx+1`, where `idx` is the highest integer such that `out/<prefix>-<idx>` exists.
    """
    if not (out / f"{prefix}-0").is_dir():
        idx = 0
    else:
        indices = [
            int(directory.name.removeprefix(f"{prefix}-"))
            for directory in out.iterdir()
            if re.match(rf"{prefix}-\d+", directory.name)
        ]
        idx = max(indices) + 1
    return idx


def _update_mammos_uppasd_data(out: pathlib.Path, label: str) -> None:
    """Update `info.yaml` of MammosUppasdData object.

    If this file does not exist, create it.
    Otherwise, add the label of this run to the `history` field.
    """
    mammos_uppasd_data = out / "info.yaml"
    if not mammos_uppasd_data.is_file():
        info_dict = {
            "metadata": {
                "mode": "mammos_uppasd_data",
            },
            "history": [
                label,
            ],
        }
    else:
        with open(mammos_uppasd_data) as f:
            info_dict = yaml.safe_load(f)
            info_dict["history"] = info_dict["history"] + [label]

    with open(mammos_uppasd_data, "w") as f:
        yaml.dump(info_dict, f)
