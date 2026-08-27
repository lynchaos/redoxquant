"""Bioanalytical assay metrics, quality control profiles, and assay range limits.

Implements standard Ligand-Binding Assay (LBA) validation calculations per
Azadeh et al. (2018), doi:10.1208/s12248-017-0159-4, and FDA/EMA bioanalytical
method validation guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import schema
from .curve import CalibrationCurve


@dataclass
class AssayLimits:
    """Dynamic range and detection limits for an assay calibration."""

    lod: float
    lloq: float
    uloq: float
    lod_signal: float
    eval_table: pd.DataFrame


@dataclass
class DilutionLinearityResult:
    """Evaluation of dilutional linearity and sample parallelism across dilution series."""

    is_linear: bool
    summary: pd.DataFrame
    details: pd.DataFrame


def relative_error(
    df: pd.DataFrame,
    curve: CalibrationCurve,
    nominal_col: str = schema.CONCENTRATION,
    signal_col: str = schema.SIGNAL_RU,
) -> pd.Series:
    """Calculate percent relative error (%RE) of back-calculated concentrations.

    %RE = 100 * (back_calculated - nominal) / nominal
    """
    signals = df[signal_col].to_numpy(dtype=float)
    nominals = df[nominal_col].to_numpy(dtype=float)
    back_calc = curve.back_calculate(signals)
    with np.errstate(invalid="ignore", divide="ignore"):
        re = 100.0 * (back_calc - nominals) / nominals
    return pd.Series(re, index=df.index, name="re_pct")


def total_error_profile(
    standards_df: pd.DataFrame,
    curve: CalibrationCurve,
    *,
    re_max_pct: float = 20.0,
    lloq_re_max_pct: float = 25.0,
    cv_max_pct: float = 20.0,
) -> pd.DataFrame:
    """Compute accuracy (%RE), precision (%CV), and Total Error (%TE) per standard level.

    Total Error = |%RE| + 2 * %CV (Azadeh et al. 2018).

    Parameters
    ----------
    standards_df:
        Canonical frame containing standard measurements.
    curve:
        Fitted calibration curve.
    re_max_pct:
        Maximum allowable absolute %RE for mid-range standards (default 20.0%).
    lloq_re_max_pct:
        Maximum allowable absolute %RE at LLOQ/ULOQ anchor points (default 25.0%).
    cv_max_pct:
        Maximum allowable %CV (default 20.0%).

    Returns
    -------
    pd.DataFrame
        Summary table by nominal concentration level.
    """
    df = standards_df.copy()
    df["back_calc_conc"] = curve.back_calculate(df[schema.SIGNAL_RU].to_numpy(dtype=float))

    grouped = df.groupby(schema.CONCENTRATION, dropna=True)
    stats_list = []
    concs_sorted = sorted(grouped.groups.keys())

    for i, conc in enumerate(concs_sorted):
        group = grouped.get_group(conc)
        n = len(group)
        calc_vals = group["back_calc_conc"].to_numpy(dtype=float)
        mean_calc = float(np.nanmean(calc_vals))
        std_calc = float(np.nanstd(calc_vals, ddof=1)) if n > 1 else 0.0

        cv_pct = (100.0 * std_calc / mean_calc) if (mean_calc > 0 and np.isfinite(mean_calc)) else float("nan")
        re_pct = (100.0 * (mean_calc - conc) / conc) if conc > 0 else float("nan")
        te_pct = (abs(re_pct) + 2.0 * cv_pct) if (np.isfinite(re_pct) and np.isfinite(cv_pct)) else float("nan")

        # Boundary levels (lowest and highest) allow relaxed LLOQ/ULOQ tolerances
        is_boundary = (i == 0) or (i == len(concs_sorted) - 1)
        allowed_re = lloq_re_max_pct if is_boundary else re_max_pct

        passed = bool(
            np.isfinite(re_pct)
            and np.isfinite(cv_pct)
            and abs(re_pct) <= allowed_re
            and cv_pct <= cv_max_pct
        )

        stats_list.append(
            {
                "nominal_concentration": conc,
                "n": n,
                "mean_back_calc": mean_calc,
                "std_back_calc": std_calc,
                "cv_pct": cv_pct,
                "re_pct": re_pct,
                "total_error_pct": te_pct,
                "passed_qc": passed,
            }
        )

    return pd.DataFrame(stats_list)


def compute_assay_limits(
    standards_df: pd.DataFrame,
    curve: CalibrationCurve,
    *,
    re_max_pct: float = 20.0,
    lloq_re_max_pct: float = 25.0,
    cv_max_pct: float = 20.0,
    blank_signal: Optional[float] = None,
    blank_sd: Optional[float] = None,
) -> AssayLimits:
    """Compute Limit of Detection (LOD), LLOQ, and ULOQ from standards and curve.

    LLOQ (Lower Limit of Quantification) is the lowest standard meeting the
    acceptance criteria (|%RE| <= lloq_re_max_pct and %CV <= cv_max_pct).
    ULOQ (Upper Limit of Quantification) is the highest standard meeting criteria.

    LOD is calculated as the concentration corresponding to the baseline signal
    plus/minus 3.3 * blank_sd (ascending vs descending). If blank parameters are
    not provided, they are estimated from the lowest concentration standard.
    """
    profile = total_error_profile(
        standards_df,
        curve,
        re_max_pct=re_max_pct,
        lloq_re_max_pct=lloq_re_max_pct,
        cv_max_pct=cv_max_pct,
    )

    passed_rows = profile[profile["passed_qc"]]
    if len(passed_rows) == 0:
        lloq = float("nan")
        uloq = float("nan")
    else:
        lloq = float(passed_rows["nominal_concentration"].min())
        uloq = float(passed_rows["nominal_concentration"].max())

    # LOD computation
    if blank_signal is None or blank_sd is None:
        # Estimate blank signal from the baseline asymptote / lowest standard
        min_conc_row = profile.sort_values("nominal_concentration").iloc[0]
        min_conc = min_conc_row["nominal_concentration"]
        subset = standards_df[standards_df[schema.CONCENTRATION] == min_conc][schema.SIGNAL_RU]
        est_signal = float(subset.mean())
        est_sd = float(subset.std(ddof=1)) if len(subset) > 1 else max(est_signal * 0.05, 1.0)
        blank_signal = blank_signal if blank_signal is not None else est_signal
        blank_sd = blank_sd if blank_sd is not None else est_sd

    if curve.descending:
        lod_sig = blank_signal - 3.3 * blank_sd
    else:
        lod_sig = blank_signal + 3.3 * blank_sd

    lod_conc = float(curve.back_calculate(lod_sig))
    if not np.isfinite(lod_conc) or lod_conc <= 0:
        # Fallback if baseline signal inverse touches asymptote
        lod_conc = lloq if np.isfinite(lloq) else float("nan")

    return AssayLimits(
        lod=lod_conc,
        lloq=lloq,
        uloq=uloq,
        lod_signal=float(lod_sig),
        eval_table=profile,
    )


def evaluate_dilution_linearity(
    df: pd.DataFrame,
    curve: CalibrationCurve,
    *,
    group_col: str = schema.TAG,
    cv_threshold_pct: float = 20.0,
    recovery_range: Tuple[float, float] = (80.0, 120.0),
) -> DilutionLinearityResult:
    """Evaluate dilutional linearity across samples prepared at multiple dilution factors.

    Calculates adjusted concentration per dilution, checks %CV of adjusted
    concentrations across dilutions, and verifies recovery percentage relative
    to the nominal/least-diluted sample.

    Parameters
    ----------
    df:
        Canonical frame with samples.
    curve:
        Calibration curve used to back-calculate concentration.
    group_col:
        Column to group sample replicate series (default TAG).
    cv_threshold_pct:
        Maximum allowable %CV across dilution series (default 20.0%).
    recovery_range:
        Acceptable recovery interval (default 80% to 120%).
    """
    work_df = df.copy()
    work_df["conc_raw"] = curve.back_calculate(work_df[schema.SIGNAL_RU].to_numpy(dtype=float))
    comp = work_df[schema.SIGNAL_COMPENSATION].to_numpy(dtype=float)
    dil = work_df[schema.DILUTION_FACTOR].to_numpy(dtype=float)
    work_df["adjusted_conc"] = curve.back_calculate(work_df[schema.SIGNAL_RU].to_numpy(dtype=float) * comp) * dil

    details_list = []
    summary_list = []
    overall_linear = True

    grouped = work_df.dropna(subset=[group_col]).groupby(group_col)
    for name, group in grouped:
        dilutions = sorted(group[schema.DILUTION_FACTOR].unique())
        if len(dilutions) < 2:
            continue

        dil_stats = []
        for d in dilutions:
            sub = group[group[schema.DILUTION_FACTOR] == d]
            mean_adj = float(sub["adjusted_conc"].mean())
            dil_stats.append((d, mean_adj))

        ref_dil, ref_adj = dil_stats[0]  # baseline (lowest dilution)
        adj_vals = [s[1] for s in dil_stats]
        mean_series = float(np.mean(adj_vals))
        std_series = float(np.std(adj_vals, ddof=1)) if len(adj_vals) > 1 else 0.0
        cv_series = (100.0 * std_series / mean_series) if mean_series > 0 else float("nan")

        for d, mean_adj in dil_stats:
            rec_pct = (100.0 * mean_adj / ref_adj) if ref_adj > 0 else float("nan")
            in_rec = recovery_range[0] <= rec_pct <= recovery_range[1]
            details_list.append(
                {
                    "group": name,
                    "dilution_factor": d,
                    "mean_adjusted_conc": mean_adj,
                    "recovery_pct": rec_pct,
                    "recovery_pass": in_rec,
                }
            )

        group_pass = bool(cv_series <= cv_threshold_pct)
        if not group_pass:
            overall_linear = False

        summary_list.append(
            {
                "group": name,
                "num_dilutions": len(dilutions),
                "mean_adjusted_conc": mean_series,
                "cv_pct_across_dilutions": cv_series,
                "dilution_linearity_pass": group_pass,
            }
        )

    summary_df = pd.DataFrame(summary_list)
    details_df = pd.DataFrame(details_list)

    return DilutionLinearityResult(
        is_linear=overall_linear and len(summary_df) > 0,
        summary=summary_df,
        details=details_df,
    )
