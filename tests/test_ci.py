"""Tests for back_calculate_with_ci (task 5.1).

Three layers:
1. Structural — BackCalcCI fields exist, lower < estimate < upper, NaN propagation.
2. Asymptote widening — CIs are demonstrably wider near the logistic limits.
3. Real-data fidelity — point estimates on the Tocilizumab fixture are unchanged.
4. Nominal coverage — delta-method CIs agree with a parametric bootstrap (self-consistency).
"""

import numpy as np
import pytest

from redoxquant import fit_calibration, schema
from redoxquant.curve import BackCalcCI, _back_calc_scalar
from redoxquant.synthetic import generate_standards


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit_noisy(seed: int = 0, cv: float = 0.03) -> object:
    std = generate_standards(seed=seed, cv=cv)
    return fit_calibration(std[schema.CONCENTRATION], std[schema.SIGNAL_RU])


# ---------------------------------------------------------------------------
# 1. Structure
# ---------------------------------------------------------------------------

def test_returns_backCalcCI_type():
    curve = _fit_noisy()
    sig = curve.predict(curve.c)  # inflection-point signal
    result = curve.back_calculate_with_ci(sig)
    assert isinstance(result, BackCalcCI)
    assert result.ci_level == 0.95


def test_scalar_input_returns_scalars():
    curve = _fit_noisy()
    sig = curve.predict(curve.c)
    result = curve.back_calculate_with_ci(sig)
    assert isinstance(result.estimate, float)
    assert isinstance(result.lower, float)
    assert isinstance(result.upper, float)


def test_array_input_returns_arrays():
    curve = _fit_noisy()
    sigs = np.array([curve.predict(c) for c in [5.0, 25.0, 80.0]])
    result = curve.back_calculate_with_ci(sigs)
    assert result.estimate.shape == (3,)
    assert result.lower.shape == (3,)
    assert result.upper.shape == (3,)


def test_intervals_bracket_estimate():
    curve = _fit_noisy(seed=1)
    # Mid-range concentrations — all should give finite, ordered CIs
    concs = np.geomspace(2.0, 80.0, 10)
    sigs = curve.predict(concs)
    result = curve.back_calculate_with_ci(sigs)
    assert np.all(np.isfinite(result.estimate))
    assert np.all(np.isfinite(result.lower))
    assert np.all(np.isfinite(result.upper))
    assert np.all(result.lower < result.estimate)
    assert np.all(result.estimate < result.upper)


def test_estimate_matches_back_calculate():
    """back_calculate_with_ci point estimate is identical to back_calculate."""
    curve = _fit_noisy(seed=2)
    sigs = curve.predict(np.geomspace(1.0, 100.0, 12))
    plain = curve.back_calculate(sigs)
    result = curve.back_calculate_with_ci(sigs)
    np.testing.assert_allclose(result.estimate, plain, rtol=1e-10)


def test_nan_outside_asymptote_band():
    """Signals outside (d, a) for a descending curve produce NaN throughout."""
    curve = _fit_noisy()
    assert curve.descending  # a > d
    outside = curve.a * 1.05  # above upper asymptote
    result = curve.back_calculate_with_ci(outside)
    assert np.isnan(result.estimate)
    assert np.isnan(result.lower)
    assert np.isnan(result.upper)


def test_raises_without_pcov():
    """CalibrationCurve with no pcov raises ValueError."""
    from redoxquant.curve import CalibrationCurve
    curve = CalibrationCurve(a=3200, b=1.2, c=25, d=450, g=1.0, r_squared=0.999)
    with pytest.raises(ValueError, match="pcov"):
        curve.back_calculate_with_ci(1000.0)


def test_wider_ci_with_higher_confidence_level():
    curve = _fit_noisy(seed=3)
    sig = curve.predict(curve.c)
    r90 = curve.back_calculate_with_ci(sig, ci=0.90)
    r99 = curve.back_calculate_with_ci(sig, ci=0.99)
    assert (r99.upper - r99.lower) > (r90.upper - r90.lower)


# ---------------------------------------------------------------------------
# 2. Asymptote widening
# ---------------------------------------------------------------------------

def test_intervals_widen_near_asymptotes():
    """Relative CI widths grow when extrapolating beyond the standard range.

    generate_standards spans 1..100 µg/mL. Concentrations well outside that
    range are extrapolations toward the logistic asymptotes — the delta method
    should report much larger *relative* uncertainty there than at the
    inflection point, which is well-supported by the fitted standards.
    """
    curve = _fit_noisy(seed=4, cv=0.03)

    # Well inside the calibrated range
    sig_mid = float(curve.predict(curve.c))

    # 10× below the minimum standard (extrapolation toward upper asymptote)
    sig_lo = float(curve.predict(0.1))
    # 10× above the maximum standard (extrapolation toward lower asymptote)
    sig_hi = float(curve.predict(1000.0))

    r_mid = curve.back_calculate_with_ci(sig_mid)
    r_lo = curve.back_calculate_with_ci(sig_lo)
    r_hi = curve.back_calculate_with_ci(sig_hi)

    assert np.isfinite(r_lo.estimate) and r_lo.estimate > 0
    assert np.isfinite(r_hi.estimate) and r_hi.estimate > 0

    rel_mid = (r_mid.upper - r_mid.lower) / r_mid.estimate
    rel_lo = (r_lo.upper - r_lo.lower) / r_lo.estimate
    rel_hi = (r_hi.upper - r_hi.lower) / r_hi.estimate

    assert rel_lo > rel_mid * 5, (
        f"CI should be much wider below calibrated range: "
        f"rel_lo={rel_lo:.3f} vs rel_mid={rel_mid:.3f}"
    )
    assert rel_hi > rel_mid * 5, (
        f"CI should be much wider above calibrated range: "
        f"rel_hi={rel_hi:.3f} vs rel_mid={rel_mid:.3f}"
    )


# ---------------------------------------------------------------------------
# 3. Real-data fidelity
# ---------------------------------------------------------------------------

def test_real_data_estimates_and_intervals(anchor_points):
    """On the Tocilizumab fixture: estimates unchanged, intervals finite and ordered."""
    x = np.array([p[0] for p in anchor_points])
    y = np.array([p[1] for p in anchor_points])
    curve = fit_calibration(x, y, model="5PL", weight="1/y2")

    sample_sigs = np.array([r[1] for r in anchor_points[:12]])
    plain = curve.back_calculate(sample_sigs)
    result = curve.back_calculate_with_ci(sample_sigs)

    np.testing.assert_allclose(result.estimate, plain, rtol=1e-10)
    assert np.all(np.isfinite(result.lower))
    assert np.all(np.isfinite(result.upper))
    assert np.all(result.lower < result.estimate)
    assert np.all(result.estimate < result.upper)


# ---------------------------------------------------------------------------
# 4. Nominal coverage (self-consistency with parametric bootstrap)
# ---------------------------------------------------------------------------

def test_nominal_coverage_via_parametric_bootstrap():
    """Delta-method 95% CI achieves ~95% coverage under the Gaussian parameter model.

    We sample parameter sets from MVN(popt, pcov) and check that the delta-method
    CI contains approximately 95% of the resulting back-calculated concentrations.
    The delta method is a linear approximation; agreement should be close for a
    mid-range signal where the inverse is nearly linear.
    """
    rng = np.random.default_rng(42)

    std = generate_standards(seed=0, cv=0.03)
    curve = fit_calibration(std[schema.CONCENTRATION], std[schema.SIGNAL_RU])

    # Pick a mid-range signal (inflection point)
    test_signal = float(curve.predict(curve.c))
    result = curve.back_calculate_with_ci(test_signal, ci=0.95)

    assert np.isfinite(result.lower) and np.isfinite(result.upper)

    # Parametric bootstrap: sample from the fitted Gaussian distribution
    free = np.array([curve.a, curve.b, curve.c, curve.d, curve.g])
    n_samples = 10_000
    samples = rng.multivariate_normal(mean=free, cov=curve.pcov, size=n_samples)

    back_concs = np.array([
        _back_calc_scalar(test_signal, p[0], p[1], p[2], p[3], p[4])
        for p in samples
    ])
    valid = back_concs[np.isfinite(back_concs)]

    coverage = float(np.mean((valid >= result.lower) & (valid <= result.upper)))
    # Allow ±6% tolerance (linear approximation error near inflection is small)
    assert 0.89 < coverage < 1.00, f"Coverage {coverage:.3f} outside [0.89, 1.00]"


def test_ascending_curve_ci():
    """CI works correctly for ascending (sandwich-assay) curves."""
    std = generate_standards(a=450.0, d=3200.0, seed=5, cv=0.03)
    curve = fit_calibration(std[schema.CONCENTRATION], std[schema.SIGNAL_RU])
    assert not curve.descending

    sigs = curve.predict(np.geomspace(2.0, 80.0, 8))
    result = curve.back_calculate_with_ci(sigs)

    assert np.all(np.isfinite(result.estimate))
    assert np.all(result.lower < result.estimate)
    assert np.all(result.estimate < result.upper)


def test_4pl_ci():
    """CI is available for 4PL-fitted curves (4×4 pcov)."""
    std = generate_standards(seed=6, cv=0.03)
    curve = fit_calibration(std[schema.CONCENTRATION], std[schema.SIGNAL_RU], model="4PL")
    assert curve.model == "4PL"
    assert curve.pcov is not None
    assert curve.pcov.shape == (4, 4)

    sig = curve.predict(curve.c)
    result = curve.back_calculate_with_ci(sig)
    assert isinstance(result.estimate, float)
    assert result.lower < result.estimate < result.upper
