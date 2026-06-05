"""Synthetic Amperia-style data, so the library is usable with zero hardware.

Generates standard curves and sample sets from a known 5PL, with optional
log-normal signal noise, so round-trip recovery can be tested and the docs can
run end to end without a real export.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema
from .curve import _five_pl


def generate_standards(
    *,
    a: float = 3200.0,
    b: float = 1.2,
    c: float = 25.0,
    d: float = 450.0,
    g: float = 1.0,
    concentrations=None,
    replicates: int = 4,
    cv: float = 0.03,
    unit: str = "µg/ml",
    seed: int | None = 0,
) -> pd.DataFrame:
    """Generate a canonical frame of standards from a known 5PL.

    Defaults describe a descending curve (a > d), matching the Tocilizumab
    demo. ``replicates`` defaults to 4, the probe count of one sensor strip.
    """
    rng = np.random.default_rng(seed)
    if concentrations is None:
        concentrations = np.geomspace(1.0, 100.0, 8)
    rows = []
    for step, conc in enumerate(concentrations, start=1):
        true_signal = float(_five_pl(conc, a, b, c, d, g))
        for _ in range(replicates):
            noisy = true_signal * float(rng.lognormal(mean=0.0, sigma=cv))
            rows.append(
                {
                    schema.STEP: step,
                    schema.DURATION_S: 33,
                    schema.DESCRIPTION: f"STD {conc:g}",
                    schema.SOLUTION_TYPE: schema.SolutionType.STANDARD.value,
                    schema.SIGNAL_RU: round(noisy, 1),
                    schema.SIGNAL_COMPENSATION: 1.0,
                    schema.CONCENTRATION: conc,
                    schema.ADJUSTED_CONCENTRATION: conc,
                    schema.UNIT: unit,
                    schema.DILUTION_FACTOR: 1.0,
                    schema.TAG: None,
                }
            )
    return pd.DataFrame(rows)[schema.CANONICAL_COLUMNS]
