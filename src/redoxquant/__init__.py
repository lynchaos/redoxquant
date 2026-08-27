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

from .bioanalysis import (
    AssayLimits,
    DilutionLinearityResult,
    compute_assay_limits,
    evaluate_dilution_linearity,
    relative_error,
    total_error_profile,
)
from .comparison import (
    BlandAltmanResult,
    DemingResult,
    PassingBablokResult,
    bland_altman,
    deming_regression,
    passing_bablok_regression,
)
from .curve import BackCalcCI, CalibrationCurve, fit_calibration
from .feedback import send_slack_alert, submit_feedback
from .io import read_csv, read_frame, read_xlsx
from .quantify import group_stats, qc_flags, quantify
from .report import generate_html_report
from . import ai, bioanalysis, comparison, feedback, ml, report, schema, synthetic

__version__ = "0.2.0"

__all__ = [
    "AssayLimits",
    "BackCalcCI",
    "BlandAltmanResult",
    "CalibrationCurve",
    "DemingResult",
    "DilutionLinearityResult",
    "PassingBablokResult",
    "bland_altman",
    "compute_assay_limits",
    "deming_regression",
    "evaluate_dilution_linearity",
    "feedback",
    "fit_calibration",
    "generate_html_report",
    "group_stats",
    "passing_bablok_regression",
    "qc_flags",
    "quantify",
    "read_csv",
    "read_frame",
    "read_xlsx",
    "relative_error",
    "send_slack_alert",
    "submit_feedback",
    "total_error_profile",

    "ai",
    "bioanalysis",
    "comparison",
    "ml",
    "report",
    "schema",
    "synthetic",
    "__version__",
]


