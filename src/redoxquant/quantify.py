"""Quantification, replicate grouping, and basic QC on canonical frames.

The compensation/dilution arithmetic mirrors the Amperia analysis export's
``Concentration`` and ``Adjusted Concentration`` columns:

    concentration          = curve.back_calculate(signal)
    adjusted_concentration = curve.back_calculate(signal * compensation) * dilution_factor

NOTE: the exact role of the signal-compensation factor is inferred from the
export layout (a multiplicative correction applied to signal before the curve).
This should be confirmed against a fuller export that includes the standard
rows; until then it is an explicit, documented assumption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema
from .curve import CalibrationCurve


def quantify(df: pd.DataFrame, curve: CalibrationCurve) -> pd.DataFrame:
    """Add curve-derived concentration columns to a canonical frame.

    Existing instrument-reported columns are preserved; the recomputed values
    are written to ``concentration_calc`` and ``adjusted_concentration_calc``
    so they can be compared against the instrument's own output.
    """
    out = df.copy()
    signal = out[schema.SIGNAL_RU].to_numpy(dtype=float)
    comp = out[schema.SIGNAL_COMPENSATION].to_numpy(dtype=float)
    dil = out[schema.DILUTION_FACTOR].to_numpy(dtype=float)

    out["concentration_calc"] = curve.back_calculate(signal)
    out["adjusted_concentration_calc"] = curve.back_calculate(signal * comp) * dil
    return out


def group_stats(df: pd.DataFrame, value: str = schema.SIGNAL_RU) -> pd.DataFrame:
    """Per-tag summary: mean, std dev and %CV of a value column.

    Mirrors the instrument's top-right summary panel (Tag / Concentration /
    Signal / Std Dev), computed across the replicate probes of each tag.
    """
    grouped = df.dropna(subset=[schema.TAG]).groupby(schema.TAG, dropna=True)
    summary = grouped[value].agg(["count", "mean", "std"]).rename(
        columns={"count": "n", "mean": "mean", "std": "std_dev"}
    )
    summary["cv_pct"] = 100.0 * summary["std_dev"] / summary["mean"]
    if schema.CONCENTRATION in df.columns:
        summary["mean_concentration"] = grouped[schema.CONCENTRATION].mean()
    return summary.reset_index()


def qc_flags(df: pd.DataFrame, *, comp_tol: float = 0.05, cv_warn_pct: float = 20.0) -> pd.DataFrame:
    """Attach simple QC flags grounded in real export fields.

    - ``flag_compensation``: signal-compensation factor deviates from 1.0 by
      more than ``comp_tol`` (default 5%).
    - ``flag_high_cv``: the row's tag group exceeds ``cv_warn_pct`` signal CV.
    """
    out = df.copy()
    comp = out[schema.SIGNAL_COMPENSATION].to_numpy(dtype=float)
    out["flag_compensation"] = np.abs(comp - 1.0) > comp_tol

    stats = group_stats(out).set_index(schema.TAG)["cv_pct"].to_dict()
    out["flag_high_cv"] = out[schema.TAG].map(
        lambda t: bool(stats.get(t, 0.0) > cv_warn_pct) if pd.notna(t) else False
    )
    return out
