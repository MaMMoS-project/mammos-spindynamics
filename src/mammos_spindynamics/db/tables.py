"""Functions for reading tables."""

from pathlib import Path
import pandas as pd
from rich import print
from scipy.interpolate import interp1d
from textwrap import dedent

DIR = Path(__file__).parent


def get_M(
    formula=None,
    spacegroup=None,
    cell_length_a=None,
    cell_length_b=None,
    cell_length_c=None,
    cell_angle_alpha=None,
    cell_angle_beta=None,
    cell_angle_gamma=None,
    cell_volume=None,
    ICSD_label=None,
    OQMD_label=None,
    interpolation_kind="linear",
):
    """Get magnetization function from table.

    This function retrieves intrinsic properties at zero temperature
    given a certain chemical formula, by looking the values
    in a database.

    :param formula: Chemical formula
    :type formula: str
    :param spacegroup: Space group
    :type spacegroup: str
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
    :raises LookupError: Requested material not found in database.
    :raises LookupError: Too many results found with this formula.
    """
    df = find_material(
        formula=formula,
        spacegroup=spacegroup,
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
    num_results = len(df)
    if num_results == 0:
        raise LookupError("Requested material not found in database.")
    elif num_results > 1:
        # list all possible choice
        error_string = (
            "Too many results. Please refine your search.\n"
            + "Avilable materials based on request:\n"
        )
        for row, material in df.iterrows():
            error_string += describe_material(material)
        raise LookupError(error_string)

    material = df.iloc[0]
    table = pd.read_csv(DIR / "data" / material.table)
    print(f"Loaded material.\n{describe_material(material)}")
    return interp1d(table["Temp"], table["Mavg"], kind=interpolation_kind)


def find_material(**kwargs):
    """Find materials in database.

    This function retrieves one or known materials from the database
    `db.csv` by filtering for any requirements given in **kwargs.

    :returns: Dataframe containing materials with requested qualities.
        Possibly empty.
    :rtype: pandas.DataFrame
    """
    df = pd.read_csv(
        DIR / "db.csv",
        dtype={
            "formula": str,
            "spacegroup": str,
            "cell_length_a": float,
            "cell_length_b": float,
            "cell_length_c": float,
            "cell_angle_alpha": float,
            "cell_angle_beta": float,
            "cell_angle_gamma": float,
            "cell_volume": float,
            "ICSD_label": str,
            "OQMD_label": str,
        },
    )
    for key, value in kwargs.items():
        if value is not None:
            df = df[df[key] == value]
    return df


def describe_material(material):
    """Describe material in a complete way.

    This function returns a string listing the properties of the given material.

    :param material: Material
    :type material: pandas.core.frame.DataFrame
    """
    return dedent(
        f"""
            Chemical Formula: {material.formula}
            Spacegroup: {material.spacegroup}
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
