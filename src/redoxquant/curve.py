"""Calibration-curve fitting for ligand-binding-style assays.

Amperia's onboard analysis applies a 5-parameter logistic (5PL) sigmoidal fit,
following the calibration best practices of Azadeh et al. (2018),
doi:10.1208/s12248-017-0159-4. This module reimplements that model so results
match the instrument's orientation, and supports both ascending (sandwich) and
descending (competitive/inhibition) curves. The Tocilizumab demo data is
descending: higher concentration yields lower signal.

Parameter convention for the 5PL:

    y = d + (a - d) / (1 + (x / c) ** b) ** g

where ``a`` is the response as x -> 0, ``d`` is the response as x -> inf,
``c`` is the inflection concentration, ``b`` the Hill slope and ``g`` the
asymmetry factor. A 4PL is the special case ``g == 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm as _norm


def _five_pl(x, a, b, c, d, g):
    x = np.asarray(x, dtype=float)
    return d + (a - d) / np.power(1.0 + np.power(x / c, b), g)


def _back_calc_scalar(sig: float, a: float, b: float, c: float, d: float, g: float) -> float:
    """Scalar 5PL inverse; returns NaN if the signal is outside the asymptote band."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (a - d) / (sig - d)
        inner = float(np.power(float(ratio), 1.0 / g)) - 1.0
        x = c * float(np.power(inner, 1.0 / b))
    return float(x) if np.isfinite(x) and x > 0 else float("nan")


@dataclass
class BackCalcCI:
    """Point estimate and confidence interval from :meth:`CalibrationCurve.back_calculate_with_ci`.

    Attributes
    ----------
    estimate:
        Back-calculated concentration(s).
    lower, upper:
        Confidence bounds at ``ci_level``.
    ci_level:
        The requested confidence level, e.g. 0.95.
    """

    estimate: Union[float, np.ndarray]
    lower: Union[float, np.ndarray]
    upper: Union[float, np.ndarray]
    ci_level: float


@dataclass
class CalibrationCurve:
    """A fitted 5PL (or 4PL) standard curve."""

    a: float
    b: float
    c: float
    d: float
    g: float
    r_squared: float
    model: str = "5PL"
    # Parameter covariance matrix from curve_fit; None when not available.
    pcov: Union[np.ndarray, None] = field(default=None, repr=False, compare=False)

    @property
    def descending(self) -> bool:
        """True when signal decreases as concentration increases."""
        return self.a > self.d

    @property
    def params(self) -> tuple[float, float, float, float, float]:
        return (self.a, self.b, self.c, self.d, self.g)

    def predict(self, concentration):
        """Forward model: concentration -> expected signal (RU)."""
        return _five_pl(concentration, *self.params)

    def back_calculate(self, signal):
        """Inverse model: signal (RU) -> concentration.

        Returns NaN for signals outside the (a, d) asymptote band, since those
        cannot be inverted on the logistic.
        """
        a, b, c, d, g = self.params
        y = np.asarray(signal, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = (a - d) / (y - d)
            inner = np.power(ratio, 1.0 / g) - 1.0
            x = c * np.power(inner, 1.0 / b)
        x = np.where(np.isfinite(x) & (x > 0), x, np.nan)
        return x if x.ndim else float(x)

    def back_calculate_with_ci(self, signal, *, ci: float = 0.95) -> BackCalcCI:
        """Back-calculate with a parameter-uncertainty confidence interval.

        Uses the delta method: propagates the parameter covariance matrix
        (``pcov``, captured at fit time) through the numerical Jacobian of the
        5PL inverse to obtain a first-order variance estimate for each
        back-calculated concentration.

        Parameters
        ----------
        signal:
            Observed signal (RU), scalar or array-like.
        ci:
            Confidence level (default 0.95 for a 95% CI).

        Returns
        -------
        BackCalcCI
            ``.estimate`` — point estimate(s), identical to
            :meth:`back_calculate`; ``.lower`` / ``.upper`` — confidence
            bounds; ``.ci_level`` — the requested level.

        Notes
        -----
        Intervals are wider near the curve asymptotes, where the logistic
        inverse has a steep Jacobian and small parameter perturbations produce
        large concentration changes.  Requires ``pcov`` to be set (guaranteed
        when the curve was produced by :func:`fit_calibration`).
        """
        if self.pcov is None:
            raise ValueError(
                "pcov is not available. Refit with fit_calibration to obtain uncertainty estimates."
            )

        signal_arr = np.asarray(signal, dtype=float)
        scalar_in = signal_arr.ndim == 0
        signal_arr = np.atleast_1d(signal_arr)

        estimate_arr = np.atleast_1d(self.back_calculate(signal_arr)).astype(float)

        z = float(_norm.ppf((1.0 + ci) / 2.0))

        # Free parameters: 4 for 4PL (a,b,c,d), 5 for 5PL (a,b,c,d,g).
        if self.model == "4PL":
            free = np.array([self.a, self.b, self.c, self.d], dtype=float)

            def _bc(p: np.ndarray, s: float) -> float:
                return _back_calc_scalar(s, p[0], p[1], p[2], p[3], 1.0)
        else:
            free = np.array([self.a, self.b, self.c, self.d, self.g], dtype=float)

            def _bc(p: np.ndarray, s: float) -> float:
                return _back_calc_scalar(s, p[0], p[1], p[2], p[3], p[4])

        n_free = len(free)
        pcov = self.pcov[:n_free, :n_free]

        lower = np.empty(len(signal_arr))
        upper = np.empty(len(signal_arr))

        pcov_ok = np.all(np.isfinite(pcov))

        for i, (sig, est) in enumerate(zip(signal_arr, estimate_arr)):
            if not np.isfinite(est) or not pcov_ok:
                lower[i] = upper[i] = np.nan
                continue

            # Central-difference Jacobian of back_calculate w.r.t. free params.
            J = np.zeros(n_free)
            for j in range(n_free):
                h = max(abs(free[j]) * 1e-5, 1e-7)
                p_p = free.copy()
                p_m = free.copy()
                p_p[j] += h
                p_m[j] -= h
                x_p = _bc(p_p, float(sig))
                x_m = _bc(p_m, float(sig))
                if np.isfinite(x_p) and np.isfinite(x_m):
                    J[j] = (x_p - x_m) / (2.0 * h)
                else:
                    J[j] = 0.0

            var_x = float(J @ pcov @ J)
            se_x = np.sqrt(max(var_x, 0.0))
            lower[i] = est - z * se_x
            upper[i] = est + z * se_x

        if scalar_in:
            return BackCalcCI(
                estimate=float(estimate_arr[0]),
                lower=float(lower[0]),
                upper=float(upper[0]),
                ci_level=ci,
            )
        return BackCalcCI(estimate=estimate_arr, lower=lower, upper=upper, ci_level=ci)


def fit_calibration(
    concentration,
    signal,
    *,
    model: str = "5PL",
    weight: str | None = "1/y2",
) -> CalibrationCurve:
    """Fit a calibration curve to standard (concentration, signal) pairs.

    Parameters
    ----------
    concentration, signal:
        Standard points. Concentrations must be > 0.
    model:
        ``"5PL"`` (default) or ``"4PL"`` (asymmetry fixed to 1).
    weight:
        ``"1/y2"`` for variance-stabilising 1/y^2 weighting (Azadeh default),
        ``"1/y"``, or ``None`` for ordinary least squares.
    """
    x = np.asarray(concentration, dtype=float)
    y = np.asarray(signal, dtype=float)
    if np.any(x <= 0):
        raise ValueError("Concentrations must be strictly positive for a logistic fit.")
    if x.size < (4 if model == "4PL" else 5):
        raise ValueError(f"{model} needs at least {4 if model == '4PL' else 5} points; got {x.size}.")

    a0, d0 = float(np.max(y)), float(np.min(y))
    c0 = float(np.exp(np.mean(np.log(x))))  # geometric mean
    p0 = [a0, 1.0, c0, d0, 1.0]
    lower = [0.0, 0.05, x.min() * 1e-3, 0.0, 0.1]
    upper = [np.inf, 20.0, x.max() * 1e3, np.inf, 10.0]

    if weight == "1/y2":
        sigma = 1.0 / np.maximum(np.abs(y), 1e-9) ** 2
    elif weight == "1/y":
        sigma = 1.0 / np.maximum(np.abs(y), 1e-9)
    else:
        sigma = None
    # curve_fit expects sigma as standard deviations; pass 1/sqrt(w).
    sigma_sd = None if sigma is None else 1.0 / np.sqrt(sigma)

    if model == "4PL":
        def f(xx, a, b, c, d):
            return _five_pl(xx, a, b, c, d, 1.0)

        popt, pcov = curve_fit(
            f, x, y, p0=p0[:4], sigma=sigma_sd, absolute_sigma=False,
            bounds=(lower[:4], upper[:4]), maxfev=200000,
        )
        a, b, c, d = popt
        g = 1.0
    else:
        popt, pcov = curve_fit(
            _five_pl, x, y, p0=p0, sigma=sigma_sd, absolute_sigma=False,
            bounds=(lower, upper), maxfev=200000,
        )
        a, b, c, d, g = popt

    resid = y - _five_pl(x, a, b, c, d, g)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return CalibrationCurve(a=a, b=b, c=c, d=d, g=g, r_squared=r2, model=model, pcov=pcov)
