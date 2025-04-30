"""Functions for reading tables."""

import pathlib
import numpy as np
import pandas as pd
from rich import print
from scipy.interpolate import interp1d
from textwrap import dedent

DATA_DIR = pathlib.Path(__file__).parent / "data"


def check_short_label(short_label):
    """Check that short label follows the standards and returns material parameters.

    :param short_label: Short label containing chemical formula and space group
        number separated by a hyphen.
    :type short_label: str
    :raises ValueError: Wrong format.
    :return: Chemical formula and space group number.
    :rtype: (str,int)
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


def get_M(
    short_label=None,
    chemical_formula=None,
    space_group_name=None,
    space_group_number=None,
    cell_length_a=None,
    cell_length_b=None,
    cell_length_c=None,
    cell_angle_alpha=None,
    cell_angle_beta=None,
    cell_angle_gamma=None,
    cell_volume=None,
    ICSD_label=None,
    OQMD_label=None,
    jfile=None,
    momfile=None,
    posfile=None,
    interpolation_kind="linear",
):
    """Get magnetization function from table.

    This function retrieves the time-dependent magnetization
    from a database of spin dynamics calculations, by querying
    material information or UppASD input files.

    :param chemical_formula: Chemical formula
    :type chemical_formula: str
    :param space_group_name: Space group name
    :type space_group_name: str
    :param space_group_number: Space group number
    :type space_group_number: int
    :param cell_length_a: Cell length x
    :type cell_length_a: float
    :param cell_length_b: Cell length y
    :type cell_length_b: float
    :param cell_length_c: Cell length z
    :type cell_length_c: float
    :param cell_angle_alpha: Cell angle alpha
    :type cell_angle_alpha: float
    :param cell_angle_beta: Cell angle beta
    :type cell_angle_beta: float
    :param cell_angle_gamma: Cell angle gamma
    :type cell_angle_gamma: float
    :param cell_volume: Cell volume
    :type cell_volume: float
    :param ICSD_label: Label in the NIST Inorganic Crystal Structure Database.
    :type ICSD_label: str
    :param OQMD_label: Label in the the Open Quantum Materials Database.
    :type OQMD_label: str
    :param interpolation_kind: attribute `kind` for `scipy.interpolate.interp1d`.
        From scipy's documentation::

            The string has to be one of `linear`, `nearest`, `nearest-up`, `zero`,
            `slinear`, `quadratic`, `cubic`, `previous`, or `next`. `zero`, `slinear`,
            `quadratic` and `cubic` refer to a spline interpolation of zeroth, first,
            second or third order; `previous` and `next` simply return the previous or
            next value of the point; `nearest-up` and `nearest` differ when
            interpolating half-integers (e.g. 0.5, 1.5) in that `nearest-up` rounds
            up and `nearest` rounds down. Default is `linear`.

    :type interpolation_kind: str
    :returns: Interpolator function based on available data.
    :rtype: scipy.interpolate.iterp1d
    """
    if posfile is not None:
        table = load_uppasd_simulation(jfile=jfile, momfile=momfile, posfile=posfile)
    else:
        table = load_ab_initio_data(
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
    return interp1d(table["T[K]"], table["M[muB]"], kind=interpolation_kind)


def load_uppasd_simulation(jfile, momfile, posfile):
    """Find UppASD simulation results with given input files in database.

    :param jfile: Location of `jfile`
    :type jfile: str or pathlib.Path
    :param momfile: Location of `momfile`.
    :type momfile: str or pathlib.Path
    :param posfile: Location of `posfile`.
    :type posfile: str or pathlib.Path
    :raises LookupError: Simulation not found in database.
    :return: Table of pre-calculated Temperature-dependent Magnetization values.
    :rtype: pandas.DataFrame
    """
    j = parse_jfile(jfile)
    mom = parse_momfile(momfile)
    pos = parse_posfile(posfile)
    for ii in (DATA_DIR).iterdir():
        if check_input_files(ii, j, mom, pos):
            table = pd.read_csv(ii / "M.csv")
            print("Found material in database.")
            print(describe_material(material_label=ii.name))
            return table
    raise LookupError("Requested simulation not found in database.")


def parse_jfile(jfile):
    """Parse jfile, input for UppASD.

    :param jfile: Location of `jfile`.
    :type jfile: str or pathlib.Path
    :returns: Dataframe of exchange interactions.
    :rtype: pandas.DataFrame
    :raises SyntaxError: Wrong formatting.
        See https://uppasd.github.io/UppASD-manual/input/#exchange
        for the correct formatting.
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


def parse_momfile(momfile):
    """Parse momfile, input for UppASD.

    :param momfile: Location of `momfile`.
    :type momfile: str or pathlib.Path
    :returns: Dictionary of magnetic moment information.
    :rtype: dict
    :raises SyntaxError: Wrong formatting.
        See https://uppasd.github.io/UppASD-manual/input/#momfile
        for the correct formatting.
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


def parse_posfile(posfile):
    """Parse posfile, input for UppASD.

    :param posfile: Location of `posfile`.
    :type posfile: str or pathlib.Path
    :returns: Dictionary of atoms position information.
    :rtype: dict
    :raises SyntaxError: Wrong formatting.
        See https://uppasd.github.io/UppASD-manual/input/#posfile
        for the correct formatting.
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


def check_input_files(dir_i, j, mom, pos):
    """Check if UppASD inputs are equivalent to the ones in directory `dir_i`.

    The extracted input information `j`, `mom`, and `pos` are compared with the
    extracted information from the files in directory `dir_i`.
    If the inputs are close enough, this function returns `True`.

    :param dir_i: Considered directory in the database
    :type dir_i: pathlib.Path
    :param j: Dataframe of exchange interactions.
    :type j: pandas.DataFrame
    :param mom: Dictionary of magnetic moment information.
    :type mom: dict
    :param pos: Dictionary of atoms position information.
    :type pos: dict
    :returns: `True` if the inputs match almost exactly. `False` otherwise.
    :rtype: bool
    """
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


def load_ab_initio_data(**kwargs):
    """Load material with given structure information.

    :raises LookupError: Requested material not found in database.
    :raises LookupError: Too many results found with this formula.
    :return: Table of pre-calculated Temperature-dependent Magnetization values.
    :rtype: pandas.DataFrame
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
    print("Found material in database.")
    print(describe_material(material))
    return pd.read_csv(DATA_DIR / material.label / "M.csv")


def find_materials(**kwargs):
    """Find materials in database.

    This function retrieves one or known materials from the database
    `db.csv` by filtering for any requirements given in **kwargs.

    :returns: Dataframe containing materials with requested qualities.
        Possibly empty.
    :rtype: pandas.DataFrame
    """
    df = pd.read_csv(
        dtype={
        DATA_DIR / "db.csv",
            "chemical_formula": str,
            "space_group_name": str,
            "space_group_number": int,
            "cell_length_a": float,
            "cell_length_b": float,
            "cell_length_c": float,
            "cell_angle_alpha": float,
            "cell_angle_beta": float,
            "cell_angle_gamma": float,
            "cell_volume": float,
            "ICSD_label": str,
            "OQMD_label": str,
            "label": str,
        },
    )
    for key, value in kwargs.items():
        if value is not None:
            df = df[df[key] == value]
    return df


def describe_material(material=None, material_label=None):
    """Describe material in a complete way.

    This function returns a string listing the properties of the given material
    or the given material label.

    :param material: Material dataframe containing structure information.
        Defaults to `None`.
    :type material: pandas.core.frame.DataFrame
    :param material_label: Label of material in local database.
        Defaults to `None`.
    :type material_label: str
    :return: Well-formatted material information.
    :rtype: str
    :raise ValueError: Material and material label cannot be both empty.
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
