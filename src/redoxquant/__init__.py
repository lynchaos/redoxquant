"""redoxquant — an open companion library for Amperia analysis exports.

Parses Amperia (Redox Electrochemical Detection) analysis exports into tidy
data, refits the instrument's 5PL standard curve (ascending or descending),
and layers on the rigour a benchtop GUI typically omits: confidence on
back-calculated concentrations, replicate CV, dilution linearity and QC.

This project works strictly downstream of the instrument export and is not
affiliated with or endorsed by Abselion / HexagonFab Ltd. "Amperia" and
"Abselion" are trademarks of their owner and are used here only to describe
file compatibility.
"""

from __future__ import annotations

from .curve import BackCalcCI, CalibrationCurve, fit_calibration
from .io import read_csv, read_frame, read_xlsx
from .quantify import group_stats, qc_flags, quantify
from . import schema, synthetic

__version__ = "0.0.1"

__all__ = [
    "BackCalcCI",
    "CalibrationCurve",
    "fit_calibration",
    "read_csv",
    "read_xlsx",
    "read_frame",
    "quantify",
    "group_stats",
    "qc_flags",
    "schema",
    "synthetic",
    "__version__",
]
