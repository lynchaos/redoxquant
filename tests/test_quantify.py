"""Quantification, grouping and QC tests, anchored on the real demo data."""

import numpy as np

from redoxquant import read_csv, fit_calibration, quantify, group_stats, qc_flags, schema


def _fitted_curve(anchor_points):
    x = np.array([p[0] for p in anchor_points])
    y = np.array([p[1] for p in anchor_points])
    return fit_calibration(x, y, model="5PL", weight="1/y2")


def test_adjusted_concentration_matches_instrument(demo_csv_path, anchor_points):
    df = read_csv(demo_csv_path)
    curve = _fitted_curve(anchor_points)
    q = quantify(df, curve)

    # On a descending curve, the +0.3% signal compensation lowers concentration,
    # so adjusted must sit just below the raw back-calculated value.
    assert (q["adjusted_concentration_calc"] < q["concentration_calc"]).all()

    # And it should track the instrument's reported adjusted concentration.
    rel = (
        (q["adjusted_concentration_calc"] - q[schema.ADJUSTED_CONCENTRATION]).abs()
        / q[schema.ADJUSTED_CONCENTRATION]
    )
    assert rel.max() < 0.01


def test_group_stats_match_reported_std_dev(demo_csv_path):
    df = read_csv(demo_csv_path)
    stats = group_stats(df).set_index(schema.TAG)
    # Reported signal Std Dev in the export summary: mid ~15.7, hi ~13.9.
    assert abs(stats.loc["mid", "std_dev"] - 15.7) < 6.0
    assert abs(stats.loc["hi", "std_dev"] - 13.9) < 6.0
    assert (stats["cv_pct"] < 5.0).all()  # tight replicate precision


def test_qc_flags_present(demo_csv_path):
    df = read_csv(demo_csv_path)
    flagged = qc_flags(df, comp_tol=0.05, cv_warn_pct=20.0)
    assert "flag_compensation" in flagged.columns
    assert "flag_high_cv" in flagged.columns
    # 1.003 compensation is within 5% tolerance, CVs are tight -> nothing flagged.
    assert not flagged["flag_compensation"].any()
    assert not flagged["flag_high_cv"].any()
