"""Tests for orthogonal method comparison (Deming, Passing-Bablok, Bland-Altman)."""

import numpy as np
import pytest

from redoxquant import bland_altman, deming_regression, passing_bablok_regression


def test_deming_regression_identity():
    # Identity relationship y = x with small noise
    rng = np.random.default_rng(42)
    x = np.linspace(10.0, 100.0, 20)
    y = x + rng.normal(0.0, 0.5, size=len(x))

    res = deming_regression(x, y)
    assert abs(res.slope - 1.0) < 0.05
    assert abs(res.intercept) < 2.0
    assert res.r_squared > 0.99
    assert res.slope_ci[0] < res.slope < res.slope_ci[1]
    assert res.intercept_ci[0] < res.intercept < res.intercept_ci[1]


def test_passing_bablok_regression_linear():
    # Known relationship: y = 2.0 * x + 5.0
    rng = np.random.default_rng(123)
    x = np.linspace(5.0, 50.0, 15)
    y = 2.0 * x + 5.0 + rng.normal(0.0, 0.2, size=len(x))

    res = passing_bablok_regression(x, y)
    assert abs(res.slope - 2.0) < 0.1
    assert abs(res.intercept - 5.0) < 1.0
    assert res.slope_ci[0] <= res.slope <= res.slope_ci[1]
    assert res.is_linear


def test_bland_altman_agreement():
    # Two methods with average bias of 2.0
    rng = np.random.default_rng(99)
    ref = np.linspace(20.0, 80.0, 25)
    test = ref + 2.0 + rng.normal(0.0, 0.5, size=len(ref))

    res = bland_altman(ref, test)
    assert abs(res.mean_difference - 2.0) < 0.3
    assert res.lower_loa < res.mean_difference < res.upper_loa
    assert len(res.differences) == 25
    assert not res.percentage_differences


def test_bland_altman_percentage():
    ref = np.array([10.0, 20.0, 30.0, 40.0])
    test = np.array([11.0, 22.0, 33.0, 44.0])  # +10% relative to ref
    res = bland_altman(ref, test, percentage=True)
    assert res.percentage_differences
    assert abs(res.mean_difference - 9.5) < 1.0
