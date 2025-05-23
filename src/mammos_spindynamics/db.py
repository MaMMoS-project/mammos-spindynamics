"""Functions for reading tables."""

from __future__ import annotations
import numpy as np
import pandas
import pandas as pd
import pathlib
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from rich import print
from textwrap import dedent

import mammos_entity as me
import mammos_units as u

DATA_DIR = pathlib.Path(__file__).parent / "data"


def check_short_label(short_label: str) -> str | int:
    """Check that short label follows the standards and returns material parameters.

    Args:
        short_label: Short label containing chemical formula and space group
    number separated by a hyphen.

    Returns:
        Chemical formula and space group number.

    Raises:
        ValueError: Wrong format.

    """
    short_label_list = short_label.split("-")
    if len(short_label_list) != 2:
        raise ValueError(
            dedent(
                """
                Wrong format for `short_label`.
                Please use the format <chemical_formula>-<space_group_number>.
                """
            )
        )
    chemical_formula = short_label_list[0]
    space_group_number = int(short_label_list[1])
    return chemical_formula, space_group_number


def get_spontaneous_magnetisation(
    short_label: str | None = None,
    chemical_formula: str | None = None,
    space_group_name: str | None = None,
    space_group_number: int | None = None,
    cell_length_a: float | None = None,
    cell_length_b: float | None = None,
    cell_length_c: float | None = None,
    cell_angle_alpha: float | None = None,
    cell_angle_beta: float | None = None,
    cell_angle_gamma: float | None = None,
    cell_volume: float | None = None,
    ICSD_label: str | None = None,
    OQMD_label: str | None = None,
    jfile=None,
    momfile=None,
    posfile=None,
    print_info: bool = False,
) -> MagnetisationData:
    """Get spontaneous magnetization interpolator from a database.

    This function retrieves the temperature-dependent spontaneous magnetization
    from a local database of spin dynamics calculations.
    Data is retrieved by querying material information or UppASD input files.

    Args:
        short_label: short label
        chemical_formula: Chemical formula
        space_group_name: Space group name
        space_group_number: Space group number
        cell_length_a: Cell length x
        cell_length_b: Cell length y
        cell_length_c: Cell length z
        cell_angle_alpha: Cell angle alpha
        cell_angle_beta: Cell angle beta
        cell_angle_gamma: Cell angle gamma
        cell_volume: Cell volume
        ICSD_label: Label in the NIST Inorganic Crystal Structure Database.
        OQMD_label: Label in the the Open Quantum Materials Database.
        jfile: TODO
        momfile: TODO
        posfile: TODO
        print_info: Whether to print information about the retrieved material.

    Returns:
        Interpolator function based on available data.

    """
    if posfile is not None:
        table = load_uppasd_simulation(
            jfile=jfile, momfile=momfile, posfile=posfile, print_info=print_info
        )
    else:
        if short_label is not None:
            chemical_formula, space_group_number = check_short_label(short_label)
        table = load_ab_initio_data(
            print_info=print_info,
            chemical_formula=chemical_formula,
            space_group_name=space_group_name,
            space_group_number=space_group_number,
            cell_length_a=cell_length_a,
            cell_length_b=cell_length_b,
            cell_length_c=cell_length_c,
            cell_angle_alpha=cell_angle_alpha,
            cell_angle_beta=cell_angle_beta,
            cell_angle_gamma=cell_angle_gamma,
            cell_volume=cell_volume,
            ICSD_label=ICSD_label,
            OQMD_label=OQMD_label,
        )

    return MagnetisationData(
        me.Entity("ThermodynamicTemperature", value=table["T[K]"], unit=u.K),
        me.Ms(table["M[A/m]"], unit=u.A / u.m),
    )


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class MagnetisationData:
    """Magnetisation data.

    Contains temperature and spontaneous magnetisation data.
    """

    T: me.Entity
    Ms: me.Entity

    @property
    def dataframe(self):
        """Dataframe containing temperature and spontaneous magnetisation data."""
        return pd.DataFrame(
            {
                "T": self.T,
                "Ms": self.Ms,
            }
        )


def load_uppasd_simulation(
    jfile: str | pathlib.Path,
    momfile: str | pathlib.Path,
    posfile: str | pathlib.Path,
    print_info: bool = False,
) -> pandas.DataFrame:
    """Find UppASD simulation results with given input files in database.

    Args:
        jfile: Location of `jfile`
        momfile: Location of `momfile`.
        posfile: Location of `posfile`.
        print_info: Whether to print information about the retrieved material.

    Returns:
        Table of pre-calculated Temperature-dependent Magnetization values.

    Raises:
        LookupError: Simulation not found in database.

    """
    j = parse_jfile(jfile)
    mom = parse_momfile(momfile)
    pos = parse_posfile(posfile)
    for ii in DATA_DIR.iterdir():
        if ii.is_dir() and check_input_files(ii, j, mom, pos):
            table = pd.read_csv(ii / "M.csv", header=[0, 1])
            if print_info:
                print("Found material in database.")
                print(describe_material(material_label=ii.name))
            return table
    raise LookupError("Requested simulation not found in database.")


def parse_jfile(jfile: str | pathlib.Path) -> pandas.DataFrame:
    """Parse jfile, input for UppASD.

    See https://uppasd.github.io/UppASD-manual/input/#exchange
    for the correct formatting.

    Args:
        jfile: Location of `jfile`.

    Returns:
        Dataframe of exchange interactions.

    Raises:
        SyntaxError: Wrong formatting.

    """
    with open(jfile) as ff:
        jlines = ff.readlines()
    try:
        df = pd.DataFrame(
            [
                [int(x) for x in li[:-1]] + [float(x) for x in li[-1:]]
                for li in [line.split() for line in jlines]
            ],
            columns=[
                "atom_i",
                "atom_j",
                "interaction_x",
                "interaction_y",
                "interaction_z",
                "exchange_energy[mRy]",
            ],
        ).sort_values(
            by=["atom_i", "atom_j", "interaction_x", "interaction_y", "interaction_z"],
        )
        return df
    except ValueError:
        raise SyntaxError(
            dedent(
                """
                Unable to parse jfile.
                Please check syntax according to
                https://uppasd.github.io/UppASD-manual/input/#exchange
                """
            )
        ) from None


def parse_momfile(momfile: str | pathlib.Path) -> dict:
    """Parse momfile, input for UppASD.

    See https://uppasd.github.io/UppASD-manual/input/#momfile
    for the correct formatting.

    Args:
        momfile: Location of `momfile`.

    Returns:
        Dictionary of magnetic moment information.

    Raises:
        SyntaxError: Wrong formatting.

    """
    with open(momfile) as ff:
        momlines = ff.readlines()
    try:
        mom = {
            int(line[0]): {
                "chemical_type": int(line[1]),
                "magnetic_moment_magnitude[muB]": float(line[2]),
                "magnetic_moment_direction": np.array([float(x) for x in line[3:]]),
            }
            for line in [ll.split() for ll in momlines]
        }
        return mom
    except ValueError:
        raise SyntaxError(
            dedent(
                """
                Unable to parse momfile.
                Please check syntax according to
                https://uppasd.github.io/UppASD-manual/input/#momfile
                """
            )
        ) from None


def parse_posfile(posfile: str | pathlib.Path) -> dict:
    """Parse posfile, input for UppASD.

    See https://uppasd.github.io/UppASD-manual/input/#posfile
    for the correct formatting.

    Args:
        posfile: Location of `posfile`.

    Returns:
        Dictionary of atoms position information.

    Raises:
        SyntaxError: Wrong formatting.

    """
    with open(posfile) as ff:
        poslines = ff.readlines()
    try:
        pos = {
            int(line[0]): {
                "atom_type": int(line[1]),
                "atom_position": np.array([float(x) for x in line[2:]]),
            }
            for line in [ll.split() for ll in poslines]
        }
        return pos
    except ValueError:
        raise SyntaxError(
            dedent(
                """
                Unable to parse posfile.
                Please check syntax according to
                https://uppasd.github.io/UppASD-manual/input/#posfile
                """
            )
        ) from None


def check_input_files(
    dir_i: pathlib.Path, j: pandas.DataFrame, mom: dict, pos: dict
) -> bool:
    """Check if UppASD inputs are equivalent to the ones in directory `dir_i`.

    The extracted input information `j`, `mom`, and `pos` are compared with the
    extracted information from the files in directory `dir_i`.
    If the inputs are close enough, this function returns `True`.

    Args:
        dir_i: Considered directory in the database
        j: Dataframe of exchange interactions.
        mom: Dictionary of magnetic moment information.
        pos: Dictionary of atoms position information.

    Returns:
        bool: True` if the inputs match almost exactly. `False` otherwise.

    """
    if not (
        (dir_i / "jfile").is_file()
        or (dir_i / "momfile").is_file()
        or (dir_i / "posfile").is_file()
    ):
        return False

    if (dir_i / "jfile").is_file():
        j_i = parse_jfile(dir_i / "jfile")
        if not j_i.drop("exchange_energy[mRy]", axis=1).equals(
            j.drop("exchange_energy[mRy]", axis=1)
        ):
            return False
        if not np.allclose(
            j_i["exchange_energy[mRy]"].to_numpy(),
            j["exchange_energy[mRy]"].to_numpy(),
        ):
            return False

    if (dir_i / "momfile").is_file():
        mom_i = parse_momfile(dir_i / "momfile")
        if len(mom_i) != len(mom):
            return False
        for index, site in mom_i.items():
            if (
                site["chemical_type"] != mom[index]["chemical_type"]
                or not np.allclose(
                    site["magnetic_moment_magnitude[muB]"],
                    mom[index]["magnetic_moment_magnitude[muB]"],
                )
                or not np.allclose(
                    site["magnetic_moment_direction"],
                    mom[index]["magnetic_moment_direction"],
                )
            ):
                return False

    if (dir_i / "posfile").is_file():
        pos_i = parse_posfile(dir_i / "posfile")
        if len(pos_i) != len(pos):
            return False
        for index, atom in pos_i.items():
            if atom["atom_type"] != pos[index]["atom_type"] or not np.allclose(
                atom["atom_position"],
                pos[index]["atom_position"],
            ):
                return False

    return True


def load_ab_initio_data(print_info: bool = False, **kwargs) -> pandas.DataFrame:
    """Load material with given structure information.

    Args:
        print_info: Print info
        **kwargs: Selection arguments

    Returns:
        Table of pre-calculated Temperature-dependent Magnetization values.

    Raises:
        LookupError: Requested material not found in database.
        LookupError: Too many results found with this formula.

    """
    df = find_materials(**kwargs)
    num_results = len(df)
    if num_results == 0:
        raise LookupError("Requested material not found in database.")
    elif num_results > 1:  # list all possible choice
        error_string = (
            "Too many results. Please refine your search.\n"
            + "Avilable materials based on request:\n"
        )
        for row, material in df.iterrows():
            error_string += describe_material(material)
        raise LookupError(error_string)

    material = df.iloc[0]
    if print_info:
        print("Found material in database.")
        print(describe_material(material))
    return pd.read_csv(DATA_DIR / material.label / "M.csv")


def find_materials(**kwargs) -> pandas.DataFrame:
    """Find materials in database.

    This function retrieves one or known materials from the database
    `db.csv` by filtering for any requirements given in **kwargs.

    Args:
        **kwargs: Selection arguments

    Returns:
        Dataframe containing materials with requested qualities. Possibly empty.

    """
    df = pd.read_csv(
        DATA_DIR / "db.csv",
        converters={
            "chemical_formula": str,
            "space_group_name": str,
            "space_group_number": int,
            "cell_length_a": u.Quantity,
            "cell_length_b": u.Quantity,
            "cell_length_c": u.Quantity,
            "cell_angle_alpha": u.Quantity,
            "cell_angle_beta": u.Quantity,
            "cell_angle_gamma": u.Quantity,
            "cell_volume": u.Quantity,
            "ICSD_label": str,
            "OQMD_label": str,
            "label": str,
        },
    )
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, u.Quantity):
                df = df[df[key] == value.to(df[key].unit)]
            else:
                df = df[df[key] == value]
    return df


def describe_material(
    material: pandas.DataFrame | None = None, material_label: str | None = None
) -> str:
    """Describe material in a complete way.

    This function returns a string listing the properties of the given material
    or the given material label.

    Args:
        material: Material dataframe containing structure information.
        material_label: Label of material in local database.

    Returns:
        Description of the material.

    Raises:
        ValueError: If material and material label are both None.

    """
    if material is None and material_label is None:
        raise ValueError("Material and material label cannot be both empty.")
    if material_label is not None:
        df = find_materials()
        material = df[df["label"] == material_label].iloc[0]
    return dedent(
        f"""
            Chemical Formula: {material.chemical_formula}
            Space group name: {material.space_group_name}
            Space group number: {material.space_group_number}
            Cell length a: {material.cell_length_a}
            Cell length b: {material.cell_length_b}
            Cell length c: {material.cell_length_c}
            Cell angle alpha: {material.cell_angle_alpha}
            Cell angle beta: {material.cell_angle_beta}
            Cell angle gamma: {material.cell_angle_gamma}
            Cell volume: {material.cell_volume}
            ICSD_label: {material.ICSD_label}
            OQMD_label: {material.OQMD_label}
            """
    )
