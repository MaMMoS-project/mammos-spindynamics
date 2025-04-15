"""Functions for reading tables."""

from pathlib import Path
import polars as pl
from rich import print
from scipy.interpolate import interp1d
from textwrap import dedent

DIR = Path(__file__).parent


def get_M(formula=None, OQMD_label=None, structure=None, spacegroup=None):
    """Get magnetization function from table.

    This function retrieves intrinsic properties at zero temperature
    given a certain chemical formula, by looking the values
    in a database.

    :param formula: Chemical formula
    :type formula: str
    :param structure: Structure type
    :type structure: str
    """
    df = pl.scan_csv(
        DIR / "db.csv",
        schema_overrides={"structure": pl.String},
    )
    if formula is not None:
        df_filtered = df.filter(pl.col("formula") == formula)
    if OQMD_label is not None:
        df_filtered = df.filter(pl.col("OQMD_label") == OQMD_label)
    if structure is not None:
        df_filtered = df.filter(pl.col("structure") == structure)
    if spacegroup is not None:
        df_filtered = df.filter(pl.col("spacegroup") == spacegroup)

    df_filtered = df_filtered.collect()
    num_results = len(df_filtered)
    if num_results == 0:
        raise LookupError("Requested material not found in database.")
    elif num_results > 1:
        raise LookupError("Too many results found with this formula.")

    material = df_filtered.row(0, named=True)
    table = pl.read_csv(DIR / material["table"])

    print(
        dedent(
            f"""
            Loaded material.
            Chemical Formula: {material['formula']}
            Structure: {material['structure']}
            Spacegroup: {material['spacegroup']}
            OQMD_label: {material['OQMD_label']}
            """
        )
    )
    return interp1d(table["Temp"], table["Mavg"])
