"""UppASD Input class."""

from __future__ import annotations

import pathlib
import re
from collections.abc import Iterable
from io import StringIO
from typing import TYPE_CHECKING, Any

import numpy as np
from jinja2 import Environment, PackageLoader, select_autoescape
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
)

if TYPE_CHECKING:
    import numpy


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


DEFAULT_VALUES = {}
#     # Geometry and composition
#     "BC": ["0", "0", "0"],
#     "C1"                = (/one,zero,zero/)
#     C2                = (/zero,one,zero/)
#     C3                = (/zero,zero,one/)
#     Landeg_glob       = 2.0_dblprec
#     set_landeg        = 0
#     NT                = 0
#     Sym               = 0
#     do_sortcoup       = 'N'
#     posfile           = 'posfile'
#     posfiletype       = 'C'
#     alat              = 1.0_dblprec
#     scalefac          = 1.0_dblprec
#     momfile           = 'momfile'
#     momfile_i         = 'momfile_i'
#     momfile_f         = 'momfile_f'
#     amp_rnd           = 0.0_dblprec
#     amp_rnd_path      = 0.0_dblprec
#     block_size        = 1
#     metatype          = 0
#     metanumb          = 0
#     relaxed_if        = 'Y'
#     fixed_if          = 'Y'

#     #Induced moment data
#     ind_mom_flag      = 'N'
#     ind_mom_type      =  1
#     renorm_coll       = 'N'
#     ind_tol           = 0.0010_dblprec

#     #Exchange data
#     maptype           = 1
#     ham_inp%exc_inter         = 'N'
#     ham_inp%map_multiple      = .false.
#     ham_inp%jij_scale         = 1.0_dblprec
#     ham_inp%ea_model          = .false.
#     ham_inp%ea_sigma          = 1.0_dblprec
#     ham_inp%ea_algo           = 'S'

#     #Anisotropy data
#     ham_inp%kfile             = 'kfile'
#     ham_inp%do_anisotropy     = 0
#     ham_inp%mult_axis         = 'N'

#     #Dzyaloshinskii-Moriya data
#     ham_inp%dmfile            = 'dmfile'
#     ham_inp%do_dm             = 0
#     ham_inp%dm_scale          = 1.0_dblprec
#     ham_inp%rdm_model         = .false.
#     ham_inp%rdm_sigma         = 1.0_dblprec
#     ham_inp%rdm_algo          = 'S'

#     #Symmetric anisotropic data
#     ham_inp%safile            = 'safile'
#     ham_inp%do_sa             = 0
#     ham_inp%sa_scale          = 1.0_dblprec

#     #Pseudo-Dipolar data
#     ham_inp%pdfile            = 'pdfile'
#     ham_inp%do_pd             = 0

#     #Biquadratic DM data
#     ham_inp%biqdmfile         = 'biqdmfile'
#     ham_inp%do_biqdm          = 0

#     #Biquadratic exchange data
#     ham_inp%bqfile            = 'bqfile'
#     ham_inp%do_bq             = 0

#     #Four-spin ring exchange data
#     ham_inp%ringfile          = 'ringfile'
#     ham_inp%do_ring           = 0

#     #Tensorial exchange (SKKR) data
#     ham_inp%do_jtensor        = 0
#     ham_inp%calc_jtensor      = .true.

#     #Dipole-dipole data
#     ham_inp%do_dip            = 0
#     ham_inp%print_dip_tensor  = 'N'
#     ham_inp%read_dipole       = 'N'
#     ham_inp%qdip_files        = 'qdip_file'

#     #Ewald summation data
#     ham_inp%do_ewald          = 'N'
#     ham_inp%Ewald_alpha       = 0.0_dblprec
#     ham_inp%KMAX              = (/0,0,0/)
#     ham_inp%RMAX              = (/0,0,0/)

#     #Parameters for energy minimization calculations
#     minalgo           = 1
#     minftol           = 0.000000001_dblprec
#     mintraj_step      = 100
#     vpodt             = 0.010_dblprec
#     vpomass           = 1.0_dblprec
#     minitrmax         = 10000000

#     #Parameters for GNEB calculations
#     initpath          = 1
#     spring            = 0.50_dblprec
#     mepftol           = 0.0010_dblprec
#     mepftol_ci        = 0.000010_dblprec
#     mepitrmax         = 10000000
#     meptraj_step      = 100
#     do_gneb           = 'Y'
#     do_gneb_ci        = 'N'
#     do_norm_rx        = 'N'
#     en_zero           = 'N'
#     prn_gneb_fields   = 'N'

#     #Parameters for Hessian calculations
#     do_hess_ini       = 'N'
#     do_hess_fin       = 'N'
#     do_hess_sp        = 'N'
#     eig_0             = 0.0000010_dblprec
#     is_afm            = 'N'

#     #Parameters for energy interpolation along the MEP
#     sample_num        = 500

#     #Simulation parameters
#     simid             = "_UppASD_"
#     Mensemble         = 1
#     tseed             = 1
#     para_rng          = .false.
#     llg               = 1
#     nstep             = 1
#     SDEalgh           = 1
#     ipSDEalgh         = -1   #< Is set to SDEalgh input by default.
#     aunits            = 'N'
#     perp              = 'N'

#     #Tasks
#     compensate_drift  = 0
#     do_prnstruct      = 0
#     do_storeham        = 0
#     do_prn_poscar     = 0
#     do_prn_elk        = 0
#     do_read_elk       = 0
#     do_hoc_debug      = 0
#     do_meminfo        = 0
#     evolveout         = 0
#     heisout           = 0
#     mompar            = 0
#     plotenergy        = 0
#     do_sparse         = 'N'
#     do_reduced        = 'N'

#     #Measurement phase
#     mode              = 'S'
#     hfield            = (/zero,zero,zero/)
#     mplambda1         = 0.050_dblprec
#     mplambda2         = zero
#     Temp              = zero
#     delta_t           = 1.0e-16
#     relaxtime         = 0.0e-16

#     #Monte Carlo
#     mcnstep           = 0
#     mcavrg_step       = 0
#     mcavrg_buff       = 0

#     #Initial phase
#     ipmode            = 'N'
#     ipnphase          = 0
#     ipmcnphase        = 1
#     iphfield          = (/zero,zero,zero/)

#     #Init mag
#     mseed             = 1
#     restartfile       = 'restart'
#     initmag           = 4
#     initexc           = 'N'
#     initconc          = 0.0_dblprec
#     initneigh         = 1
#     initimp           = 0.0_dblprec
#     theta0            = zero
#     phi0              = zero
#     mavg0             = -1.0_dblprec
#     roteul            = 0
#     rotang            = (/zero,zero,zero/)
#     initrotang        = 0.0_dblprec
#     initpropvec       = (/zero,zero,zero/)
#     initrotvec        = (/one,zero,zero/)
#     do_mom_legacy     = 'N'

#     #De-magnetization field
#     demag             = 'N'
#     demag1            = 'N'
#     demag2            = 'N'
#     demag3            = 'N'
#     demagvol          = zero

#     #Magnetic field pulse
#     do_bpulse         = 0
#     bpulsefile        = 'bpulsefile'
#     locfield          = 'N'

#     # LSF
#     conf_num          = 1
#     gsconf_num        = 1
#     lsf_metric        = 1
#     do_lsf            = 'N'
#     lsffile           = 'lsffile'
#     lsf_interpolate   = 'Y'
#     lsf_field         = 'T'
#     lsf_window        = 0.050_dblprec

#     #Measurements
#     logsamp           = 'N'
#     real_time_measure = 'N'
#     do_spintemp       = 'N'
#     spintemp_step     = 100

#     #Random alloy data
#     do_ralloy         = 0
#     nchmax            = 1

#     # Random number transform
#     ziggurat          = 'Y'
#     rngpol            = 'N'

#     # GPU
#     gpu_mode          = 0
#     gpu_rng           = 0
#     gpu_rng_seed      = 0

#     # I/O OVF
#     prn_ovf           = 'N'
#     read_ovf          = 'N'

#     # multi
#     do_multiscale      =.false.
#     do_prnmultiscale    =.false.
#     multiscale_old_format  ='N'
# }


class Input(BaseModel):
    """UppASD Input class.

    Further information about parameters:
    https://uppasd.github.io/UppASD-manual/input/#inpsd-dat-keywords
    https://github.com/UppASD/UppASD/blob/master/source/Input/inputdata.f90
    """

    # System parameters
    simid: str | None = None
    """8 character long simulation id. All output files will include the simid as a
    label."""
    cell: np.ndarray
    """The three lattice vectors describing the cell."""
    ncell: list[int] | None = Field(default=None, min_length=3, max_length=3)
    """Number of repetitions of the cell in each of the lattice vector directions."""
    bc: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("bc", "BC"),
        min_length=3,
        max_length=3,
    )
    """Boundary conditions (P=periodic, 0=free)."""
    posfile: pathlib.Path | None = None
    """Location of external file for the positions of the atoms in one cell, with the
    site number and type of the atom.."""
    momfile: pathlib.Path | None = None
    """Location of external file describing the magnitudes and directions of magnetic
    moments."""
    posfiletype: str | None = Field(default=None, pattern="^[CD]$")
    """Flag to change between C=Cartesian or D=direct coordinates in posfile."""

    # Hamiltonian parameters
    exchange: pathlib.Path | None = None
    """Location of external file for Heisenberg exchange couplings."""

    # General simulation parameters
    do_ralloy: int | None = None
    """Flag to set if a random alloy is being simulated (0=off/1=on)."""
    mensemble: int | None = Field(
        default=None, validation_alias=AliasChoices("mensemble", "Mensemble")
    )
    """Number of ensembles to simulate. The default value is 1, but this may be
    increased to improve statistics, especially if investigating laterally confined
    systems, such as finite clusters or other low-dimensional systems."""

    # Initialization parameters
    restartfile: pathlib.Path | None = None
    """External file containing stored snapshot from previous simulation (used when
    `initmag=4`). The format coincides with the format of the output file
    `restart.simid.out`."""
    initmag: int | None = Field(
        default=None, gt=1, lt=4, validation_alias=AliasChoices("initmag", "Initmag")
    )
    """Switch for setting up the initial configuration of the magnetic moments (1=Random
    distribution, 2=Cone, 3=aligned along direction defined in momfile, 4=Read from
    restartfile)."""

    # Initial phase parameters
    # ip_mode
    # ip_temp
    # ip_hfield
    # ip_mcnstep
    # ip_damping
    # ip_nphase
    # ip_mcanneal

    # Measurement phase parameters
    mode: str | None = Field(default=None, pattern="^[SMH]$")
    """Mode for measurement phase run (S=SD, M=Monte Carlo, H=Heat bath Monte Carlo)."""
    temp: float = Field(validation_alias=AliasChoices("temp", "T", "Temp", "TEMP"))
    """Temperature for measurement phase."""
    # hfield
    mcnstep: int | None = Field(
        default=None, validation_alias=AliasChoices("mcnstep", "mcNstep")
    )
    """Number of Monte Carlo sweeps (MCS) over the system if mode=M or H."""
    # damping
    # timestep
    # relaxtime
    # setp_bpulse

    # Parameters for measuring observables
    alat: float | None = None
    """Lattice constant (in m) for calculation of exchange stiffness."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    @field_validator("cell", mode="plain")
    @classmethod
    def _validate_cell(cls, cell: Any) -> numpy.ndarray:
        """Check if cell has right dimensions."""
        try:
            cell = np.asarray(cell)
        except ValueError as err:
            raise ValueError(
                f"Cell {cell} is not convertible to numpy.ndarray: {err}"
            ) from None
        if cell.shape != (3, 3):
            raise ValueError(f"Cell {cell} has shape {cell.shape} instead of (3, 3).")
        return cell

    @field_serializer("cell", mode="plain")
    def _serialize_cell(self, cell: numpy.ndarray) -> str:
        """Serialize cell."""
        out = " " * 5
        out += "   ".join(f"{j:z< .8f}" for j in cell[0])
        for i in [1, 2]:
            out += "\n" + " " * 10 + "   ".join(f"{j:z< .8f}" for j in cell[i])
        return out

    @field_validator("exchange", "momfile", "posfile", "restartfile", mode="after")
    @classmethod
    def _validate_path(cls, file: Any) -> pathlib.Path:
        """Check if file exists at given path."""
        file = pathlib.Path(file).resolve()
        if not file.is_file():
            raise FileNotFoundError(f"File not found at {file}.")
        return file

    @field_validator("initmag", mode="after")
    @classmethod
    def _initmag_validate(cls, initmag: int, info: ValidationInfo) -> int:
        """Check if initmag int is well defined."""
        if initmag == 4 and info.data["restartfile"] is None:
            raise ValueError("initmag=4 requires an input restartfile.")
        return initmag

    @field_serializer("bc", "ncell", mode="plain")
    def _serialize_1d_array(self, v: Iterable) -> str:
        """Serialize 1d array."""
        return " ".join(str(i) for i in v)

    def write(self, out: pathlib.Path | str) -> None:
        """Write inpsd.dat input file."""
        env = Environment(
            loader=PackageLoader("mammos_spindynamics"),
            autoescape=select_autoescape(),
        )
        template = env.get_template("inpsd.jinja")
        with open(out, "w") as file:
            file.write(
                template.render(
                    parameters=self.model_dump(exclude_unset=True),
                )
            )


def parse_inpsd_file(inpsd_file: pathlib.Path | str) -> dict:
    """Parse inpsd.dat file."""
    with open(inpsd_file, encoding="utf-8") as file:
        lines = file.readlines()
    parameters = {}
    for i, line in enumerate(lines):
        for par in STRING_PARAMETERS:
            if re.match(f"{par}.*", line):
                parameters[par] = line.split()[1]
        for par in INT_PARAMETERS:
            if re.match(f"{par}*", line):
                try:
                    parameters[par] = int(line.split()[1])
                except ValueError:
                    continue
        for par in FLOAT_PARAMETERS:
            if re.match(f"{par}*", line):
                try:
                    parameters[par] = float(line.split()[1])
                except ValueError:
                    continue
        if re.match("(bc|BC).*", line):
            parameters["bc"] = line.removeprefix("bc").split()[:3]
        if re.match("cell.*", line):
            a = np.genfromtxt(StringIO(line.removeprefix("cell")))
            b = np.genfromtxt(StringIO(lines[i + 1]))
            c = np.genfromtxt(StringIO(lines[i + 2]))
            parameters["cell"] = np.vstack((a, b, c))
        if re.match("ncell*", line):
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
