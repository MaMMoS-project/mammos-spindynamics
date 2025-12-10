import shutil
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from mammos_spindynamics import uppasd
from mammos_spindynamics.uppasd import _inpsd_dat, _simulation


def test_serialise_cell():
    assert (
        _inpsd_dat._serialise_cell([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        == "1 2 3\n      4 5 6\n      7 8 9"
    )
    assert (
        _inpsd_dat._serialise_cell([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])
        == "1.0 2.0 3.0\n      4.0 5.0 6.0\n      7.0 8.0 9.0"
    )

    with pytest.raises(ValueError):
        _inpsd_dat._serialise_cell([1, 2, 3])

    with pytest.raises(TypeError):
        _inpsd_dat._serialise_cell("abc")


def test_serialise_ncell():
    assert _inpsd_dat._serialise_ncell([12, 20, 1]) == "12 20 1"
    with pytest.raises(ValueError):
        _inpsd_dat._serialise_ncell([12, 20])

    with pytest.raises(TypeError):
        _inpsd_dat._serialise_ncell([12, 1, 20.0])


@pytest.mark.skip(reason="not yet implemented")
def test_create_input_files():
    pass


@pytest.mark.skip(reason="not yet implemented")
def test_preprocess_inpsd_dat():
    pass


def test_create_run_dir(tmp_path: Path):
    """Create a new directory for a run.

    If runs exist in the directory the index is increased
    """
    path, index = _simulation._create_run_dir(tmp_path)
    assert path == tmp_path / "run-0"
    assert index == 0

    path, index = _simulation._create_run_dir(tmp_path)
    assert path == tmp_path / "run-1"
    assert index == 1

    path, index = _simulation._create_run_dir(tmp_path / "out")
    assert path == tmp_path / "out" / "run-0"
    assert index == 0

    (tmp_path / "out" / "run-1").touch()

    path, index = _simulation._create_run_dir(tmp_path / "out")
    assert path == tmp_path / "out" / "run-2"
    assert index == 2

    (tmp_path / "a_file").touch()
    with pytest.raises(RuntimeError):
        path, index = _simulation._create_run_dir(tmp_path / "a_file")


def test_write_inputs(tmp_path: Path):
    """Write all inputs to existing run directory.

    - a new file inpsd.dat is created with content inp_file_content
    - all files in 'files_to_copy' are copied to the run dir, the new file name is the
      dict key
    - a new file mammos_spindynamics.yaml is created with content metadata
    """
    inp_file_content = "the\ninp file"
    files_to_copy = {
        "jfile": tmp_path / "jfile",
        "momfile": tmp_path / "common" / "momfile",
    }
    metadata = {"fake_data": True}

    run_path = tmp_path / "out" / "run-0"

    # output directory does not exist; created in a prior function
    with pytest.raises(FileNotFoundError):
        _simulation._write_inputs(run_path, inp_file_content, files_to_copy, metadata)

    run_path.mkdir(parents=True)

    # input files missing
    with pytest.raises(FileNotFoundError):
        _simulation._write_inputs(run_path, inp_file_content, files_to_copy, metadata)

    for path in files_to_copy.values():
        path.parent.mkdir(exist_ok=True)
        path.touch()

    _simulation._write_inputs(run_path, inp_file_content, files_to_copy, metadata)
    assert (run_path / "inpsd.dat").read_text() == "the\ninp file"
    assert (run_path / "jfile").exists()
    assert (run_path / "momfile").exists()
    assert (run_path / "mammos_spindynamics.yaml").exists()
    with open(run_path / "mammos_spindynamics.yaml") as f:
        assert yaml.safe_load(f) == metadata


def test_find_executable():
    exe = _simulation._find_executable("uppasd")
    assert isinstance(exe, Path)
    assert ".pixi/envs/default/" in exe.as_posix()  # common part on Windows and Unix
    with pytest.raises(RuntimeError):
        _simulation._find_executable("this-is-not-uppasd")


@pytest.mark.skip(reason="not yet implemented")
def test_update_metadata_file():
    pass


# === Integration tests using public interface ===


def test_simulation_run(tmp_path: Path):
    data_dir = Path(uppasd.__file__).parent.parent / "data" / "0001"
    sim = uppasd.Simulation()

    with pytest.raises(RuntimeError, match="parameters are missing"):
        sim.run(out=tmp_path)

    sim = uppasd.Simulation(
        cell=[(1.0, -0.5, 0), (0, 0.866, 0), (0, 0, 3.228)],
        posfiletype="D",
        maptype=2,
        initmag=4,
        alat=2.65e-10,
        ip_mcnstep=100,
        mcnstep=100,
        exchange=data_dir / "jfile",
        momfile=data_dir / "momfile",
    )

    with pytest.raises(RuntimeError, match="parameters are missing"):
        sim.run(out=tmp_path)

    with pytest.raises(AttributeError, match=r"file posfile not passed"):
        sim.run(out=tmp_path, ip_temp=50, temp=70)

    with pytest.raises(AttributeError, match="restartfile for initmag 4 missing"):
        sim.run(
            out=tmp_path,
            ip_temp=50,
            temp=70,
            posfile=data_dir / "posfile",
        )

    sim.run(
        out=tmp_path,
        ip_temp=50,
        temp=70,
        posfile=data_dir / "posfile",
        initmag=3,
    )
    # TODO check simulation results once the run method provides a return value


@pytest.mark.timeout(30)
def test_simulation_run_with_modified_inpsd_dat(tmp_path: Path):
    """Start simulation from an inpsd.dat file.

    The file sets a large number of MC steps, which are overwritten in run. If the
    overwrite would not work, we would run into the 30 second timeout.
    """
    # input file for Co2Fe2H4 taken from https://github.com/mammos-project/uppsala-data
    # - number of cells reduced to speed up simulation
    # - initmag placeholder replaced with 3
    (tmp_path / "input").write_text(
        dedent(
            """\
            simid UppASD__123

            cell  1.000000000000000  -0.500000000182990   0.000000000000000
                0.000000000000000   0.866011964524435   0.000000000000000
                0.000000000000000   0.000000000000000   3.228114436804486

            ncell 12  12  12
            bc    P P P
            sym 0
            posfile ./posfile
            posfiletype D

            initmag 3
            momfile ./momfile
            maptype 2
            exchange ./jfile
            ip_mode M
            ip_temp TEMP
            ip_mcnstep 25000

            mode M
            temp TEMP
            mcnstep 50000
            plotenergy 1
            do_proj_avrg Y
            do_cumu Y

            alat 2.6489381562e-10"""
        )
    )

    data_dir = Path(uppasd.__file__).parent.parent / "data" / "0001"
    sim = uppasd.Simulation(
        inpsd_dat=tmp_path / "input",
        exchange=data_dir / "jfile",
        momfile=data_dir / "momfile",
        posfile=data_dir / "posfile",
    )

    # temperature is still set to the placeholder; uppasd prints warnings but still
    # runs a simulation and returns 0; we only warn that we have found ERROR in stdout;
    # the full stdout is shown to the user in the warning message
    with pytest.warns(UserWarning, match="UppASD output contains ERROR lines"):
        sim.run(tmp_path / "my" / "output", ip_mcnstep=100, mcnstep=100)

    sim.run(tmp_path / "my" / "output", ip_mcnstep=100, mcnstep=100, T=10)
    # TODO check simulation results once the run method provides a return value


def test_simulation_run_with_unmodified_inpsd_dat(tmp_path: Path):
    """Run simulation for an inpsd.dat file without any modifications.

    Without *any* modifications is actually not true, the code will always rewrite
    the lines for posfile, momfile, and exchange to guarantee that the run directory
    is fully self-contained.
    """
    (tmp_path / "inpsd.dat").write_text(
        dedent(
            """\
            simid UppASD__123

            cell  1.000000000000000  -0.500000000182990   0.000000000000000
                0.000000000000000   0.866011964524435   0.000000000000000
                0.000000000000000   0.000000000000000   3.228114436804486

            ncell 12  12  12
            bc    P P P
            sym 0
            posfile ./posfile
            posfiletype D

            initmag 3
            momfile ./momfile
            maptype 2
            exchange ./jfile
            ip_mode M
            ip_temp 20
            ip_mcnstep 120

            mode M
            temp 20
            mcnstep 120
            plotenergy 1
            do_proj_avrg Y
            do_cumu Y

            alat 2.6489381562e-10"""
        )
    )
    data_dir = Path(uppasd.__file__).parent.parent / "data" / "0001"

    for file in ["jfile", "momfile", "posfile"]:
        shutil.copy(data_dir / file, file)
    sim = uppasd.Simulation(
        inpsd_dat=tmp_path / "inpsd.dat",
    )
    sim.run(tmp_path / "simulation-output")
    # TODO check simulation results once the run method provides a return value
