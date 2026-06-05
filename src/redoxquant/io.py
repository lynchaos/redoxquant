"""Readers that turn an Amperia analysis export into a tidy canonical frame.

The instrument writes display-formatted values such as ``x1.003`` for the
signal-compensation factor and ``39.0 µg/ml`` for concentration. These helpers
parse those into clean numeric columns plus a separate unit column, mapping the
display headers onto the canonical names in :mod:`redoxquant.schema`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import schema

# Map instrument display headers -> canonical column names.
_HEADER_MAP = {
    "step": schema.STEP,
    "duration": schema.DURATION_S,
    "description": schema.DESCRIPTION,
    "solution type": schema.SOLUTION_TYPE,
    "signal": schema.SIGNAL_RU,
    "signal compensation": schema.SIGNAL_COMPENSATION,
    "concentration": schema.CONCENTRATION,
    "adjusted concentration": schema.ADJUSTED_CONCENTRATION,
    "dilution factor": schema.DILUTION_FACTOR,
    "tags": schema.TAG,
    "tag": schema.TAG,
}

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _duration_to_seconds(value) -> int:
    """Parse ``HH:MM:SS`` (or ``MM:SS``) into total seconds."""
    if pd.isna(value):
        return 0
    parts = [int(p) for p in str(value).strip().split(":")]
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def _parse_compensation(value) -> float:
    """``x1.003`` -> ``1.003``; bare numbers pass through; blanks -> 1.0."""
    if pd.isna(value) or str(value).strip() == "":
        return 1.0
    m = _NUM_RE.search(str(value))
    return float(m.group()) if m else 1.0


def _parse_value_unit(value) -> tuple[float | None, str | None]:
    """``39.0 µg/ml`` -> ``(39.0, 'µg/ml')``; blanks -> ``(None, None)``."""
    if pd.isna(value) or str(value).strip() == "":
        return None, None
    text = str(value).strip()
    m = _NUM_RE.search(text)
    if not m:
        return None, None
    number = float(m.group())
    unit = text[m.end():].strip() or None
    return number, unit


def _normalise_headers(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in _HEADER_MAP:
            renamed[col] = _HEADER_MAP[key]
    return df.rename(columns=renamed)


def _build_canonical(df: pd.DataFrame) -> pd.DataFrame:
    df = _normalise_headers(df)

    out = pd.DataFrame()
    out[schema.STEP] = pd.to_numeric(df.get(schema.STEP), errors="coerce").astype("Int64")
    out[schema.DURATION_S] = df.get(schema.DURATION_S, pd.Series(dtype=object)).map(_duration_to_seconds)
    out[schema.DESCRIPTION] = df.get(schema.DESCRIPTION, pd.Series(dtype=object)).astype("string")
    out[schema.SOLUTION_TYPE] = (
        df.get(schema.SOLUTION_TYPE, pd.Series(dtype=object))
        .map(lambda v: schema.SolutionType.coerce(v).value if pd.notna(v) else None)
        .astype("string")
    )
    out[schema.SIGNAL_RU] = pd.to_numeric(df.get(schema.SIGNAL_RU), errors="coerce")
    out[schema.SIGNAL_COMPENSATION] = df.get(
        schema.SIGNAL_COMPENSATION, pd.Series(dtype=object)
    ).map(_parse_compensation)

    conc = df.get(schema.CONCENTRATION, pd.Series(dtype=object)).map(_parse_value_unit)
    adj = df.get(schema.ADJUSTED_CONCENTRATION, pd.Series(dtype=object)).map(_parse_value_unit)
    out[schema.CONCENTRATION] = [c[0] for c in conc]
    out[schema.ADJUSTED_CONCENTRATION] = [a[0] for a in adj]
    # Unit is taken from concentration, falling back to adjusted concentration.
    out[schema.UNIT] = [c[1] or a[1] for c, a in zip(conc, adj)]

    out[schema.DILUTION_FACTOR] = pd.to_numeric(
        df.get(schema.DILUTION_FACTOR), errors="coerce"
    ).fillna(1.0)
    out[schema.TAG] = df.get(schema.TAG, pd.Series(dtype=object)).astype("string")

    return out[schema.CANONICAL_COLUMNS]


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read an Amperia CSV export into a tidy canonical DataFrame."""
    raw = pd.read_csv(path)
    return _build_canonical(raw)


def read_xlsx(path: str | Path, sheet_name: int | str = 0) -> pd.DataFrame:
    """Read an Amperia XLSX export into a tidy canonical DataFrame."""
    raw = pd.read_excel(path, sheet_name=sheet_name)
    return _build_canonical(raw)


def read_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an in-memory frame of display-formatted values to canonical form."""
    return _build_canonical(df.copy())
