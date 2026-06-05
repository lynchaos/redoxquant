"""Canonical data model for Amperia analysis-stage exports.

Every field here is grounded in a column visible in the Amperia analysis
export (the "Quantification" view). The library works exclusively downstream
of the instrument's export: it never touches raw electrode signals or any
proprietary signal-to-RU step. Signal is consumed in Response Units (RU) as
the instrument reports it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SolutionType(str, Enum):
    """Well/role types as labelled in the Amperia software."""

    SAMPLE = "Sample"
    STANDARD = "Standard"
    SUBSTRATE = "Substrate"

    @classmethod
    def coerce(cls, value: str) -> "SolutionType":
        v = str(value).strip().lower()
        for member in cls:
            if member.value.lower() == v:
                return member
        raise ValueError(f"Unknown solution type: {value!r}")


# Canonical column names used internally throughout the library.
# The reader maps the instrument's (display) headers onto these.
STEP = "step"
DURATION_S = "duration_s"
DESCRIPTION = "description"
SOLUTION_TYPE = "solution_type"
SIGNAL_RU = "signal_ru"
SIGNAL_COMPENSATION = "signal_compensation"
CONCENTRATION = "concentration"
ADJUSTED_CONCENTRATION = "adjusted_concentration"
UNIT = "unit"
DILUTION_FACTOR = "dilution_factor"
TAG = "tag"

CANONICAL_COLUMNS = [
    STEP,
    DURATION_S,
    DESCRIPTION,
    SOLUTION_TYPE,
    SIGNAL_RU,
    SIGNAL_COMPENSATION,
    CONCENTRATION,
    ADJUSTED_CONCENTRATION,
    UNIT,
    DILUTION_FACTOR,
    TAG,
]

# Recognised concentration units in Amperia (assay-type dependent).
KNOWN_UNITS = {"µg/ml", "ug/ml", "µg/mL", "ug/mL", "vp/mL", "vp/ml"}


@dataclass(frozen=True)
class Measurement:
    """A single probe reading from one sequence step."""

    step: int
    duration_s: int
    description: str
    solution_type: SolutionType
    signal_ru: float
    signal_compensation: float
    concentration: float | None
    adjusted_concentration: float | None
    unit: str | None
    dilution_factor: float
    tag: str | None
