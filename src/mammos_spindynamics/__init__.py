"""Spindynamics-based magnetic material properties."""

import importlib.metadata

from mammos_spindynamics import db, uppasd

__all__ = ["db", "uppasd"]

__version__ = importlib.metadata.version(__package__)
