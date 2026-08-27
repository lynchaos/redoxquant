"""Tests for bioanalysis module (LOD, LLOQ, ULOQ, %RE, Total Error, Dilution Linearity)."""

import numpy as np
import pandas as pd
import pytest

from redoxquant import (
    compute_assay_limits,
    evaluate_dilution_linearity,
    fit_calibration,
    relative_error,
    schema,
    total_error_profile,
)
from redoxquant.synthetic import generate_standards


def test_relative_error_synthetic():
    std = generate_standards(seed=42, cv=0.0)  # noise-free
    curve = fit_calibration(std[schema.CONCENTRATION], std[schema.SIGNAL_RU])
    re = relative_error(std, curve)
    assert len(re) == len(std)
    assert re.abs().max() < 0.5  # within 0.5%


def test_total_error_profile_structure():
    std = generate_standards(seed=10, cv=0.03, replicates=4)
    curve = fit_calibration(std[schema.CONCENTRATION], std[schema.SIGNAL_RU])
    profile = total_error_profile(std, curve)

    assert "nominal_concentration" in profile.columns
    assert "cv_pct" in profile.columns
    assert "re_pct" in profile.columns
    assert "total_error_pct" in profile.columns
    assert "passed_qc" in profile.columns
    assert len(profile) == 8  # 8 standard concentrations
    # Total error = |%RE| + 2 * %CV
    for _, row in profile.iterrows():
        np.testing.assert_allclose(row["total_error_pct"], abs(row["re_pct"]) + 2.0 * row["cv_pct"], rtol=1e-5)


def test_compute_assay_limits_noise_free():
    std = generate_standards(seed=12, cv=0.0, replicates=4)
    curve = fit_calibration(std[schema.CONCENTRATION], std[schema.SIGNAL_RU])
    limits = compute_assay_limits(std, curve)

    assert np.isfinite(limits.lloq)
    assert np.isfinite(limits.uloq)
    assert limits.lloq == 1.0  # lowest nominal concentration
    assert limits.uloq == 100.0  # highest nominal concentration
    assert np.isfinite(limits.lod)
    assert limits.lod > 0


def test_compute_assay_limits_with_noise():
    std = generate_standards(seed=12, cv=0.02, replicates=4)
    curve = fit_calibration(std[schema.CONCENTRATION], std[schema.SIGNAL_RU])
    limits = compute_assay_limits(std, curve)

    assert np.isfinite(limits.lloq)
    assert np.isfinite(limits.uloq)
    assert limits.lloq <= limits.uloq
    assert limits.lloq >= 1.0  # LLOQ is identified within the calibrated range
    assert limits.uloq <= 100.0
    assert np.isfinite(limits.lod)


def test_evaluate_dilution_linearity_clean():
    # Build synthetic sample with perfect dilutional linearity (neat conc = 40.0)
    std = generate_standards(seed=0, cv=0.0)
    curve = fit_calibration(std[schema.CONCENTRATION], std[schema.SIGNAL_RU])

    # Sample diluted 1x (conc=40), 2x (conc=20), 4x (conc=10)
    dilutions = [1.0, 2.0, 4.0]
    true_conc = 40.0
    rows = []
    for d in dilutions:
        c_in_well = true_conc / d
        sig = float(curve.predict(c_in_well))
        rows.append(
            {
                schema.STEP: 10,
                schema.DURATION_S: 33,
                schema.DESCRIPTION: "Dilution Sample",
                schema.SOLUTION_TYPE: schema.SolutionType.SAMPLE.value,
                schema.SIGNAL_RU: sig,
                schema.SIGNAL_COMPENSATION: 1.0,
                schema.CONCENTRATION: None,
                schema.ADJUSTED_CONCENTRATION: None,
                schema.UNIT: "µg/ml",
                schema.DILUTION_FACTOR: d,
                schema.TAG: "mAb_Sample_1",
            }
        )
    df = pd.DataFrame(rows)[schema.CANONICAL_COLUMNS]

    result = evaluate_dilution_linearity(df, curve)
    assert result.is_linear
    assert len(result.summary) == 1
    assert result.summary.iloc[0]["dilution_linearity_pass"]
    assert result.summary.iloc[0]["cv_pct_across_dilutions"] < 2.0
    assert (result.details["recovery_pass"]).all()
