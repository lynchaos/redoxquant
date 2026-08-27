"""Tests for cross-method neural/regression surrogate module."""

import numpy as np
import pytest

from redoxquant.ml.surrogate import MethodBridge, SurrogatePrediction


def test_method_bridge_fit_predict():
    # Synthetic relationship: ELISA_OD = 0.001 * Amperia_RU + 0.1
    rng = np.random.default_rng(42)
    amperia_ru = np.linspace(500.0, 3500.0, 30)
    elisa_od = 0.001 * amperia_ru + 0.1 + rng.normal(0.0, 0.02, size=len(amperia_ru))

    bridge = MethodBridge(target_assay_name="ELISA_OD450", degree=1)
    bridge.fit(amperia_ru, elisa_od)

    test_ru = np.array([1000.0, 2000.0, 3000.0])
    pred = bridge.predict(test_ru)

    assert isinstance(pred, SurrogatePrediction)
    assert len(pred.estimate) == 3
    assert np.all(pred.lower < pred.estimate)
    assert np.all(pred.estimate < pred.upper)
    assert abs(pred.estimate[1] - (0.001 * 2000.0 + 0.1)) < 0.05
