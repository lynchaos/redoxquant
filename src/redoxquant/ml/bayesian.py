"""Hierarchical Bayesian 5PL calibration and few-shot standard curve fitting.

Enables accurate 5PL standard curve fitting with as few as 2–3 standard points
by combining observation likelihood with prior parameter distributions learned
from historical assay runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm as _norm

from ..curve import BackCalcCI, CalibrationCurve, _back_calc_scalar, _five_pl


@dataclass
class Bayesian5PLPrior:
    """Prior distribution over log-transformed 5PL parameters [ln_a, ln_b, ln_c, ln_d, ln_g]."""

    mu: np.ndarray  # shape (5,)
    cov: np.ndarray  # shape (5, 5)

    @classmethod
    def default_descending(cls) -> "Bayesian5PLPrior":
        """Generic prior for descending (competitive / inhibition) Amperia assays."""
        # a ~ 3500, b ~ 1.1, c ~ 20.0, d ~ 450, g ~ 1.0
        mu = np.array([np.log(3500.0), np.log(1.1), np.log(20.0), np.log(450.0), np.log(1.0)])
        # Broad log-space variance: (0.4)^2 on asymptotes, (0.3)^2 on slope/inflection
        cov = np.diag([0.25, 0.15, 0.25, 0.25, 0.15])
        return cls(mu=mu, cov=cov)

    @classmethod
    def default_ascending(cls) -> "Bayesian5PLPrior":
        """Generic prior for ascending (sandwich) Amperia assays."""
        # a ~ 450, b ~ 1.1, c ~ 20.0, d ~ 3500, g ~ 1.0
        mu = np.array([np.log(450.0), np.log(1.1), np.log(20.0), np.log(3500.0), np.log(1.0)])
        cov = np.diag([0.25, 0.15, 0.25, 0.25, 0.15])
        return cls(mu=mu, cov=cov)

    @classmethod
    def from_historical_curves(cls, curves: List[CalibrationCurve]) -> "Bayesian5PLPrior":
        """Fit empirical Bayes prior from a collection of historical calibration curves."""
        if len(curves) < 2:
            raise ValueError("Need at least 2 historical curves to compute empirical prior covariance.")
        param_mat = np.array([
            [np.log(max(c.a, 1e-3)), np.log(max(c.b, 1e-3)), np.log(max(c.c, 1e-3)),
             np.log(max(c.d, 1e-3)), np.log(max(c.g, 1e-3))]
            for c in curves
        ])
        mu = np.mean(param_mat, axis=0)
        cov = np.cov(param_mat, rowvar=False) + np.eye(5) * 1e-4  # regularized
        return cls(mu=mu, cov=cov)


@dataclass
class BayesianCalibrationCurve:
    """A fitted Bayesian 5PL standard curve with posterior covariance."""

    a: float
    b: float
    c: float
    d: float
    g: float
    r_squared: float
    prior: Bayesian5PLPrior
    posterior_cov_log: np.ndarray  # covariance in log-param space

    @property
    def descending(self) -> bool:
        return self.a > self.d

    @property
    def params(self) -> Tuple[float, float, float, float, float]:
        return (self.a, self.b, self.c, self.d, self.g)

    def predict(self, concentration):
        return _five_pl(concentration, *self.params)

    def back_calculate(self, signal):
        a, b, c, d, g = self.params
        y = np.asarray(signal, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = (a - d) / (y - d)
            inner = np.power(ratio, 1.0 / g) - 1.0
            x = c * np.power(inner, 1.0 / b)
        x = np.where(np.isfinite(x) & (x > 0), x, np.nan)
        return x if x.ndim else float(x)

    def sample_posterior(self, n: int = 1000, seed: Optional[int] = None) -> np.ndarray:
        """Sample n parameter sets [a, b, c, d, g] from the posterior distribution."""
        rng = np.random.default_rng(seed)
        log_mean = np.array([np.log(self.a), np.log(self.b), np.log(self.c), np.log(self.d), np.log(self.g)])
        samples_log = rng.multivariate_normal(mean=log_mean, cov=self.posterior_cov_log, size=n)
        return np.exp(samples_log)

    def back_calculate_with_credible_interval(
        self,
        signal,
        *,
        credible_interval: float = 0.95,
        n_samples: int = 2000,
        seed: int = 42,
    ) -> BackCalcCI:
        """Compute point estimate and Bayesian Credible Interval for back-calculated concentration."""
        samples = self.sample_posterior(n=n_samples, seed=seed)
        signal_arr = np.atleast_1d(np.asarray(signal, dtype=float))
        scalar_in = np.ndim(signal) == 0

        estimates = self.back_calculate(signal_arr)
        lower = np.empty(len(signal_arr))
        upper = np.empty(len(signal_arr))

        alpha = (1.0 - credible_interval) / 2.0
        q_lo = alpha * 100.0
        q_hi = (1.0 - alpha) * 100.0

        for i, sig in enumerate(signal_arr):
            back_vals = np.array([
                _back_calc_scalar(float(sig), p[0], p[1], p[2], p[3], p[4])
                for p in samples
            ])
            valid = back_vals[np.isfinite(back_vals) & (back_vals > 0)]
            if len(valid) > n_samples * 0.2:
                lower[i] = float(np.percentile(valid, q_lo))
                upper[i] = float(np.percentile(valid, q_hi))
            else:
                lower[i] = upper[i] = np.nan

        if scalar_in:
            return BackCalcCI(
                estimate=float(estimates[0]),
                lower=float(lower[0]),
                upper=float(upper[0]),
                ci_level=credible_interval,
            )
        return BackCalcCI(
            estimate=np.asarray(estimates),
            lower=lower,
            upper=upper,
            ci_level=credible_interval,
        )


def fit_bayesian_5pl(
    concentration,
    signal,
    *,
    prior: Optional[Bayesian5PLPrior] = None,
    weight: str = "1/y2",
) -> BayesianCalibrationCurve:
    """Fit a Bayesian 5PL calibration curve using Maximum A Posteriori (MAP) inference.

    Parameters
    ----------
    concentration, signal:
        Calibration standard points. Can be as few as 2–3 points when using an informative prior.
    prior:
        Optional Bayesian5PLPrior; defaults to generic descending prior if unspecified.
    weight:
        Likelihood weighting ('1/y2', '1/y', or None).
    """
    x = np.asarray(concentration, dtype=float)
    y = np.asarray(signal, dtype=float)

    if np.any(x <= 0):
        raise ValueError("Concentrations must be strictly positive.")
    if len(x) < 2:
        raise ValueError("Bayesian 5PL requires at least 2 calibration points.")

    if prior is None:
        # Determine orientation from endpoints
        if y[np.argmax(x)] < y[np.argmin(x)]:
            prior = Bayesian5PLPrior.default_descending()
        else:
            prior = Bayesian5PLPrior.default_ascending()

    inv_cov_prior = np.linalg.pinv(prior.cov)

    # Observation weights (sigma standard deviations)
    if weight == "1/y2":
        sigma_sq = (np.maximum(np.abs(y), 1e-9) * 0.05) ** 2  # 5% relative SD
    elif weight == "1/y":
        sigma_sq = np.maximum(np.abs(y), 1e-9) * 0.5
    else:
        sigma_sq = np.ones_like(y) * (np.var(y) if len(y) > 1 else 100.0)

    # Negative log posterior objective function: -ln P(theta | D) = -ln P(D | theta) - ln P(theta)
    def nll_map(theta_log: np.ndarray) -> float:
        a = np.exp(theta_log[0])
        b = np.exp(theta_log[1])
        c = np.exp(theta_log[2])
        d = np.exp(theta_log[3])
        g = np.exp(theta_log[4])

        y_pred = _five_pl(x, a, b, c, d, g)
        if np.any(~np.isfinite(y_pred)):
            return 1e12

        # Data log-likelihood term
        ll_data = -0.5 * np.sum(((y - y_pred) ** 2) / sigma_sq)

        # Prior log-likelihood term
        diff_prior = theta_log - prior.mu
        ll_prior = -0.5 * float(diff_prior @ inv_cov_prior @ diff_prior)

        return -(ll_data + ll_prior)

    # Initial guess
    p0_log = prior.mu.copy()
    res = minimize(
        nll_map,
        p0_log,
        method="L-BFGS-B",
        bounds=[
            (np.log(10.0), np.log(50000.0)),
            (np.log(0.1), np.log(10.0)),
            (np.log(1e-3), np.log(1e5)),
            (np.log(10.0), np.log(50000.0)),
            (np.log(0.1), np.log(10.0)),
        ],
    )

    opt_theta_log = res.x
    opt_a, opt_b, opt_c, opt_d, opt_g = np.exp(opt_theta_log)

    # Approximate posterior covariance by numerical Hessian at MAP
    eps = 1e-4
    H = np.zeros((5, 5))
    f0 = nll_map(opt_theta_log)
    for i in range(5):
        for j in range(5):
            t_ij = opt_theta_log.copy()
            t_ij[i] += eps
            t_ij[j] += eps
            f_ij = nll_map(t_ij)

            t_i = opt_theta_log.copy()
            t_i[i] += eps
            f_i = nll_map(t_i)

            t_j = opt_theta_log.copy()
            t_j[j] += eps
            f_j = nll_map(t_j)

            H[i, j] = (f_ij - f_i - f_j + f0) / (eps * eps)

    H_sym = 0.5 * (H + H.T)
    try:
        post_cov = np.linalg.inv(H_sym)
        # Check positive definiteness
        if np.any(np.linalg.eigvals(post_cov) <= 0):
            post_cov = np.linalg.pinv(H_sym + np.eye(5) * 1e-4)
    except Exception:
        post_cov = prior.cov * 0.5

    # R^2 calculation
    y_pred_opt = _five_pl(x, opt_a, opt_b, opt_c, opt_d, opt_g)
    ss_res = float(np.sum((y - y_pred_opt) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if len(y) > 1 else 1.0
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    return BayesianCalibrationCurve(
        a=float(opt_a),
        b=float(opt_b),
        c=float(opt_c),
        d=float(opt_d),
        g=float(opt_g),
        r_squared=float(r2),
        prior=prior,
        posterior_cov_log=post_cov,
    )
