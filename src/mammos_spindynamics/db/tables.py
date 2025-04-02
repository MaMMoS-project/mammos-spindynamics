"""Functions for reading tables."""

import inspect
from pathlib import Path
import polars as pl
from scipy.interpolate import interp1d

DIR = Path(__file__).parent


def get_M(formula="", OQMD_label="", structure="", spacegroup=""):
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
    if formula != "":
        df_filtered = df.filter(pl.col("formula") == formula)
    if OQMD_label != "":
        df_filtered = df.filter(pl.col("OQMD_label") == str(OQMD_label))
    if structure != "":
        df_filtered = df.filter(pl.col("structure") == str(structure))
    if spacegroup != "":
        df_filtered = df.filter(pl.col("spacegroup") == str(spacegroup))

    df_filtered = df_filtered.collect()
    num_results = len(df_filtered)
    if num_results == 0:
        raise LookupError("Requested material not found in database.")
    elif num_results > 1:
        raise LookupError("Too many results found with this formula.")

    material = df_filtered.row(0, named=True)
    table = pl.read_csv(DIR / material["table"])

    print(
        inspect.cleandoc(
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
