"""Orthogonal method comparison statistics (ELISA, BLI, SPR, HPLC vs Amperia).

Implements CLSI EP09-A3 standard method comparison algorithms:
- Deming regression (accounting for measurement error in both X and Y methods)
- Passing–Bablok non-parametric regression (robust to outliers and non-normality)
- Bland–Altman agreement analysis (mean bias and 95% limits of agreement)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import stats


@dataclass
class DemingResult:
    """Results from Deming orthogonal linear regression."""

    slope: float
    intercept: float
    slope_ci: Tuple[float, float]
    intercept_ci: Tuple[float, float]
    r_squared: float
    lambda_ratio: float


@dataclass
class PassingBablokResult:
    """Results from Passing–Bablok non-parametric regression."""

    slope: float
    intercept: float
    slope_ci: Tuple[float, float]
    intercept_ci: Tuple[float, float]
    cusum_p_value: float
    is_linear: bool


@dataclass
class BlandAltmanResult:
    """Results from Bland–Altman agreement analysis."""

    mean_difference: float
    sd_difference: float
    lower_loa: float
    upper_loa: float
    percentage_differences: bool
    means: np.ndarray
    differences: np.ndarray


def deming_regression(
    x,
    y,
    *,
    lambda_ratio: float = 1.0,
    alpha: float = 0.05,
) -> DemingResult:
    """Perform Deming linear regression between two analytical methods.

    Parameters
    ----------
    x, y:
        Paired measurements from the reference and test methods.
    lambda_ratio:
        Ratio of variance of error in X to variance of error in Y (default 1.0).
    alpha:
        Significance level for confidence intervals (default 0.05 for 95% CI).

    Returns
    -------
    DemingResult
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]
    n = len(x_clean)

    if n < 3:
        raise ValueError(f"Deming regression requires at least 3 points, got {n}.")

    x_bar = float(np.mean(x_clean))
    y_bar = float(np.mean(y_clean))

    s_xx = float(np.sum((x_clean - x_bar) ** 2))
    s_yy = float(np.sum((y_clean - y_bar) ** 2))
    s_xy = float(np.sum((x_clean - x_bar) * (y_clean - y_bar)))

    if abs(s_xy) < 1e-12:
        # Perfectly flat or un-correlated
        slope = 0.0
        intercept = y_bar
        return DemingResult(
            slope=slope,
            intercept=intercept,
            slope_ci=(0.0, 0.0),
            intercept_ci=(intercept, intercept),
            r_squared=0.0,
            lambda_ratio=lambda_ratio,
        )

    # Deming slope formula
    term = lambda_ratio * s_yy - s_xx
    slope = float((term + np.sqrt(term ** 2 + 4.0 * lambda_ratio * (s_xy ** 2))) / (2.0 * lambda_ratio * s_xy))
    intercept = float(y_bar - slope * x_bar)

    # Pearson correlation coefficient
    corr = s_xy / np.sqrt(s_xx * s_yy) if (s_xx > 0 and s_yy > 0) else 0.0
    r_squared = float(corr ** 2)

    # Jackknife variance estimation for confidence intervals
    slopes_jack = np.empty(n)
    intercepts_jack = np.empty(n)
    for i in range(n):
        idx = np.ones(n, dtype=bool)
        idx[i] = False
        xj, yj = x_clean[idx], y_clean[idx]
        xb, yb = float(np.mean(xj)), float(np.mean(yj))
        sxx_j = float(np.sum((xj - xb) ** 2))
        syy_j = float(np.sum((yj - yb) ** 2))
        sxy_j = float(np.sum((xj - xb) * (yj - yb)))
        if abs(sxy_j) > 1e-12:
            tj = lambda_ratio * syy_j - sxx_j
            sj = float((tj + np.sqrt(tj ** 2 + 4.0 * lambda_ratio * (sxy_j ** 2))) / (2.0 * lambda_ratio * sxy_j))
        else:
            sj = slope
        slopes_jack[i] = sj
        intercepts_jack[i] = yb - sj * xb

    se_slope = float(np.sqrt(((n - 1) / n) * np.sum((slopes_jack - np.mean(slopes_jack)) ** 2)))
    se_intercept = float(np.sqrt(((n - 1) / n) * np.sum((intercepts_jack - np.mean(intercepts_jack)) ** 2)))

    t_val = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 2))
    slope_ci = (float(slope - t_val * se_slope), float(slope + t_val * se_slope))
    intercept_ci = (float(intercept - t_val * se_intercept), float(intercept + t_val * se_intercept))

    return DemingResult(
        slope=slope,
        intercept=intercept,
        slope_ci=slope_ci,
        intercept_ci=intercept_ci,
        r_squared=r_squared,
        lambda_ratio=lambda_ratio,
    )


def passing_bablok_regression(
    x,
    y,
    *,
    alpha: float = 0.05,
) -> PassingBablokResult:
    """Perform Passing–Bablok non-parametric linear regression (CLSI EP09).

    Parameters
    ----------
    x, y:
        Paired measurements from the reference and test methods.
    alpha:
        Significance level for confidence intervals (default 0.05 for 95% CI).

    Returns
    -------
    PassingBablokResult
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]
    n = len(x_clean)

    if n < 4:
        raise ValueError(f"Passing-Bablok regression requires at least 4 points, got {n}.")

    # Calculate all pairwise slopes S_ij
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x_clean[j] - x_clean[i]
            dy = y_clean[j] - y_clean[i]
            if abs(dx) > 1e-12:
                s = dy / dx
                if not np.isclose(s, -1.0):
                    slopes.append(s)

    slopes = np.sort(np.asarray(slopes, dtype=float))
    N = len(slopes)
    if N == 0:
        raise ValueError("Could not compute valid pairwise slopes.")

    # Number of slopes < -1
    K = int(np.sum(slopes < -1.0))

    # Median index with shift K
    if N % 2 == 1:
        med_idx = (N + 1) // 2 + K - 1
    else:
        med_idx = N // 2 + K - 1

    med_idx = int(np.clip(med_idx, 0, N - 1))
    slope = float(slopes[med_idx])

    # Intercepts: median of y_i - slope * x_i
    intercepts = y_clean - slope * x_clean
    intercept = float(np.median(intercepts))

    # Confidence intervals for slope via rank statistics
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    C_gamma = z * np.sqrt(n * (n - 1) * (2 * n + 5) / 18.0)
    m1 = int(np.floor((N - C_gamma) / 2.0)) + K
    m2 = int(np.ceil((N + C_gamma) / 2.0)) + 1 + K

    m1 = int(np.clip(m1, 0, N - 1))
    m2 = int(np.clip(m2, 0, N - 1))

    slope_ci = (float(slopes[m1]), float(slopes[m2]))

    # Confidence interval for intercept
    ic_lower = float(np.median(y_clean - slope_ci[1] * x_clean))
    ic_upper = float(np.median(y_clean - slope_ci[0] * x_clean))
    intercept_ci = (min(ic_lower, ic_upper), max(ic_lower, ic_upper))

    # Linearity test via cumulative sum (CUSUM)
    residuals = y_clean - (slope * x_clean + intercept)
    signs = np.sign(residuals)
    signs[signs == 0] = 1.0
    cusum = np.cumsum(signs)
    max_cusum = float(np.max(np.abs(cusum)))

    # Kolmogorov-Smirnov style approximation for CUSUM p-value
    stat = max_cusum / np.sqrt(n)
    p_value = float(2.0 * np.exp(-2.0 * (stat ** 2)))
    p_value = min(max(p_value, 0.0), 1.0)
    is_linear = bool(p_value > 0.05)

    return PassingBablokResult(
        slope=slope,
        intercept=intercept,
        slope_ci=slope_ci,
        intercept_ci=intercept_ci,
        cusum_p_value=p_value,
        is_linear=is_linear,
    )


def bland_altman(
    x,
    y,
    *,
    percentage: bool = False,
    alpha: float = 0.05,
) -> BlandAltmanResult:
    """Compute Bland–Altman agreement metrics between two measurement methods.

    Parameters
    ----------
    x:
        Reference method values.
    y:
        Test method values (e.g. Amperia).
    percentage:
        If True, differences are expressed as percentage of the pair mean.
    alpha:
        Significance level for limits of agreement (default 0.05 for 95% LoA).

    Returns
    -------
    BlandAltmanResult
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]

    if len(x_clean) < 2:
        raise ValueError(f"Bland-Altman requires at least 2 points, got {len(x_clean)}.")

    means = (x_clean + y_clean) / 2.0
    if percentage:
        with np.errstate(invalid="ignore", divide="ignore"):
            diffs = 100.0 * (y_clean - x_clean) / means
    else:
        diffs = y_clean - x_clean

    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    lower_loa = float(mean_diff - z * sd_diff)
    upper_loa = float(mean_diff + z * sd_diff)

    return BlandAltmanResult(
        mean_difference=mean_diff,
        sd_difference=sd_diff,
        lower_loa=lower_loa,
        upper_loa=upper_loa,
        percentage_differences=percentage,
        means=means,
        differences=diffs,
    )
