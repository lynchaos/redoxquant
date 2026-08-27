"""Cross-Method Neural / Regression Surrogate (Amperia ↔ ELISA / HPLC / Octet).

Predicts equivalent legacy assay readouts (e.g. ELISA OD450, Octet BLI nm,
or Protein A HPLC peak area) directly from Amperia signals and dilution factors
with predictive uncertainty bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from scipy import stats


@dataclass
class SurrogatePrediction:
    """Predicted response in the target orthogonal method."""

    estimate: Union[float, np.ndarray]
    lower: Union[float, np.ndarray]
    upper: Union[float, np.ndarray]
    std_err: Union[float, np.ndarray]


class MethodBridge:
    """Surrogate bridge linking Amperia electrochemical signals to orthogonal bioassays."""

    def __init__(self, *, target_assay_name: str = "ELISA", degree: int = 2) -> None:
        self.target_assay_name = target_assay_name
        self.degree = degree
        self.weights_: Optional[np.ndarray] = None
        self.cov_weights_: Optional[np.ndarray] = None
        self.residual_var_: float = 0.0

    def _make_features(self, signals: np.ndarray, dilutions: Optional[np.ndarray] = None) -> np.ndarray:
        s = np.asarray(signals, dtype=float).ravel()
        if dilutions is None:
            d = np.ones_like(s)
        else:
            d = np.asarray(dilutions, dtype=float).ravel()

        cols = [np.ones_like(s)]
        for p in range(1, self.degree + 1):
            cols.append(s ** p)
            cols.append(d ** p)
            cols.append((s * d) ** p)
        return np.column_stack(cols)

    def fit(
        self,
        amperia_signals,
        target_values,
        dilution_factors=None,
        *,
        regularization: float = 1e-3,
    ) -> "MethodBridge":
        """Fit the surrogate mapping from Amperia measurements to target assay values."""
        X = self._make_features(amperia_signals, dilution_factors)
        y = np.asarray(target_values, dtype=float).ravel()

        n, p = X.shape
        if n < p:
            raise ValueError(f"Need at least {p} paired samples to train surrogate; got {n}.")

        # Ridge regularized least squares: w = (X^T X + lambda I)^-1 X^T y
        A = X.T @ X + np.eye(p) * regularization
        w = np.linalg.solve(A, X.T @ y)
        self.weights_ = w

        residuals = y - X @ w
        self.residual_var_ = float(np.sum(residuals ** 2) / max(n - p, 1))
        self.cov_weights_ = self.residual_var_ * np.linalg.pinv(A)

        return self

    def predict(
        self,
        amperia_signals,
        dilution_factors=None,
        *,
        alpha: float = 0.05,
    ) -> SurrogatePrediction:
        """Predict orthogonal assay values with prediction intervals."""
        if self.weights_ is None or self.cov_weights_ is None:
            raise ValueError("Surrogate must be fit before calling predict.")

        scalar_in = np.ndim(amperia_signals) == 0
        X = self._make_features(amperia_signals, dilution_factors)

        pred = X @ self.weights_

        # Prediction variance = sigma^2_residual + x * Cov(w) * x^T
        pred_var = np.array([self.residual_var_ + float(x_i @ self.cov_weights_ @ x_i) for x_i in X])
        std_err = np.sqrt(np.maximum(pred_var, 0.0))

        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        lower = pred - z * std_err
        upper = pred + z * std_err

        if scalar_in:
            return SurrogatePrediction(
                estimate=float(pred[0]),
                lower=float(lower[0]),
                upper=float(upper[0]),
                std_err=float(std_err[0]),
            )

        return SurrogatePrediction(
            estimate=pred,
            lower=lower,
            upper=upper,
            std_err=std_err,
        )
