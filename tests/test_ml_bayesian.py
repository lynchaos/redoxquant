"""Tests for Bayesian 5PL calibration and few-shot fitting."""

import numpy as np
import pytest

from redoxquant import schema
from redoxquant.ml.bayesian import Bayesian5PLPrior, fit_bayesian_5pl
from redoxquant.synthetic import generate_standards


def test_bayesian_5pl_few_shot_fitting():
    # True curve parameters: descending, a=3200, b=1.2, c=25, d=450, g=1.0
    full_std = generate_standards(seed=1, cv=0.01, replicates=4)

    # Use only 3 standard points (few-shot calibration)
    few_shot = full_std[full_std[schema.CONCENTRATION].isin([1.0, 25.0, 100.0])]

    prior = Bayesian5PLPrior.default_descending()
    curve = fit_bayesian_5pl(
        few_shot[schema.CONCENTRATION],
        few_shot[schema.SIGNAL_RU],
        prior=prior,
    )

    assert curve.descending
    assert curve.r_squared > 0.98

    # Test back-calculation on mid-range signal
    mid_signal = curve.predict(20.0)
    conc_recovered = curve.back_calculate(mid_signal)
    assert abs(conc_recovered - 20.0) < 3.0  # within 15% error from just 3 points


def test_bayesian_credible_intervals():
    std = generate_standards(seed=2, cv=0.02, replicates=4)
    curve = fit_bayesian_5pl(std[schema.CONCENTRATION], std[schema.SIGNAL_RU])

    test_signal = curve.predict(25.0)
    ci = curve.back_calculate_with_credible_interval(test_signal, credible_interval=0.95)

    assert isinstance(ci.estimate, float)
    assert np.isfinite(ci.lower)
    assert np.isfinite(ci.upper)
    assert ci.lower < ci.estimate < ci.upper
