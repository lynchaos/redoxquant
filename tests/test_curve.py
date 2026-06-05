"""Calibration-curve tests.

Two kinds of test:

1. Synthetic round-trip — proves the 5PL forward/inverse math is correct for
   both ascending and descending curves.
2. Real-data fidelity — refits the descending curve from the actual Amperia
   demo points and shows back-calculation recovers the instrument's reported
   concentrations. This is the credibility centrepiece: it demonstrates the
   library's inverse agrees with the instrument's own forward model on real
   numbers, not synthetic ones.
"""

import numpy as np

from redoxquant.curve import fit_calibration
from redoxquant.synthetic import generate_standards
from redoxquant import schema


def test_descending_roundtrip_synthetic():
    std = generate_standards(seed=1, cv=0.0)  # noise-free
    curve = fit_calibration(std[schema.CONCENTRATION], std[schema.SIGNAL_RU])
    assert curve.descending
    assert curve.r_squared > 0.999
    rec = curve.back_calculate(std[schema.SIGNAL_RU].to_numpy())
    rel = np.abs(rec - std[schema.CONCENTRATION].to_numpy()) / std[schema.CONCENTRATION].to_numpy()
    assert rel.max() < 1e-3  # limited by 0.1 RU export rounding


def test_ascending_roundtrip_synthetic():
    # Swap asymptotes to make an ascending (sandwich) curve.
    std = generate_standards(a=450.0, d=3200.0, seed=2, cv=0.0)
    curve = fit_calibration(std[schema.CONCENTRATION], std[schema.SIGNAL_RU])
    assert not curve.descending
    rec = curve.back_calculate(std[schema.SIGNAL_RU].to_numpy())
    rel = np.abs(rec - std[schema.CONCENTRATION].to_numpy()) / std[schema.CONCENTRATION].to_numpy()
    assert rel.max() < 1e-3  # limited by 0.1 RU export rounding


def test_real_data_fidelity(anchor_points):
    """Refit the instrument's curve and recover its reported concentrations."""
    x = np.array([p[0] for p in anchor_points])
    y = np.array([p[1] for p in anchor_points])
    curve = fit_calibration(x, y, model="5PL", weight="1/y2")

    assert curve.descending
    assert curve.r_squared > 0.99

    # The 12 sample rows: recover reported concentration from reported signal.
    sample = anchor_points[:12]
    sig = np.array([r[1] for r in sample])
    reported_conc = np.array([r[0] for r in sample])
    recovered = curve.back_calculate(sig)
    rel_err = np.abs(recovered - reported_conc) / reported_conc
    assert rel_err.max() < 0.01  # within 1% of the instrument's own numbers


def test_4pl_also_fits_real_data(anchor_points):
    x = np.array([p[0] for p in anchor_points])
    y = np.array([p[1] for p in anchor_points])
    curve = fit_calibration(x, y, model="4PL")
    assert curve.model == "4PL"
    assert curve.r_squared > 0.99
