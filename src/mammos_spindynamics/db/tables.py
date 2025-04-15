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
    :param structure: Structure type
    :type structure: str
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
    print(f"Loaded material.\n{describe_material}")
    return interp1d(table["Temp"], table["Mavg"], kind=interpolation_kind)


def find_material(
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
):
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
    if formula is not None:
        df = df[df["formula"] == formula]
    if spacegroup is not None:
        df = df[df["spacegroup"] == spacegroup]
    if cell_length_a is not None:
        df = df[df["cell_length_a"] == cell_length_a]
    if cell_length_b is not None:
        df = df[df["cell_length_b"] == cell_length_b]
    if cell_length_c is not None:
        df = df[df["cell_length_c"] == cell_length_c]
    if cell_angle_alpha is not None:
        df = df[df["cell_angle_alpha"] == cell_angle_alpha]
    if cell_angle_beta is not None:
        df = df[df["cell_angle_beta"] == cell_angle_beta]
    if cell_angle_gamma is not None:
        df = df[df["cell_angle_gamma"] == cell_angle_gamma]
    if cell_volume is not None:
        df = df[df["cell_volume"] == cell_volume]
    if ICSD_label is not None:
        df = df[df["ICSD_label"] == ICSD_label]
    if OQMD_label is not None:
        df = df[df["OQMD_label"] == OQMD_label]
    return df


def describe_material(material):
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
