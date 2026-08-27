"""Intelligent Sensor Anomaly & Fault Detection for Amperia Redox measurements.

Detects corrupted readings (microfluidic air bubbles, electrode fouling,
pipetting errors, sensor strip compensation anomalies) before curve fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd

from .. import schema


class AnomalyType(str, Enum):
    """Classification of electrochemical sensor abnormalities."""

    NORMAL = "Normal"
    BUBBLE_ARTIFACT = "Bubble Artifact"
    ELECTRODE_FOULING = "Electrode Fouling"
    PIPETTING_OUTLIER = "Pipetting Outlier"
    COMPENSATION_ANOMALY = "Compensation Anomaly"


@dataclass
class AnomalyReport:
    """Detailed anomaly diagnostic output."""

    anomalies: pd.DataFrame
    num_anomalies: int
    cleaned_df: pd.DataFrame

    @property
    def has_anomalies(self) -> bool:
        return self.num_anomalies > 0


class SensorAnomalyDetector:
    """Detector for electrochemical probe abnormalities and measurement artifacts."""

    def __init__(
        self,
        *,
        replicate_z_thresh: float = 2.5,
        compensation_tol: float = 0.05,
        min_replicates_for_stats: int = 3,
    ) -> None:
        self.replicate_z_thresh = replicate_z_thresh
        self.compensation_tol = compensation_tol
        self.min_replicates = min_replicates_for_stats

    def detect(self, df: pd.DataFrame) -> AnomalyReport:
        """Scan a canonical DataFrame and classify any anomalous probe measurements."""
        work = df.copy().reset_index(drop=True)
        n_rows = len(work)

        anomaly_types: List[AnomalyType] = [AnomalyType.NORMAL] * n_rows
        anomaly_scores: List[float] = [0.0] * n_rows
        recommendations: List[str] = ["Valid reading"] * n_rows

        # 1. Step-level replicate dispersion check (detect bubbles / single-probe spikes)
        for step_val, group in work.groupby(schema.STEP):
            if len(group) >= self.min_replicates:
                signals = group[schema.SIGNAL_RU].to_numpy(dtype=float)
                median_sig = float(np.nanmedian(signals))
                mad = float(np.nanmedian(np.abs(signals - median_sig)))
                # Robust standard deviation: 1.4826 * MAD with relative floor (at least 3% of signal)
                robust_sd = max(1.4826 * mad, abs(median_sig) * 0.03, 1.0)

                for idx in group.index:
                    sig = work.at[idx, schema.SIGNAL_RU]
                    if pd.notna(sig):
                        z_score = abs(sig - median_sig) / robust_sd
                        if z_score > self.replicate_z_thresh:
                            # If individual probe is drastically different from siblings in same step
                            anomaly_types[idx] = AnomalyType.BUBBLE_ARTIFACT
                            anomaly_scores[idx] = float(z_score)
                            diff_pct = 100.0 * (sig - median_sig) / median_sig
                            recommendations[idx] = (
                                f"Probe signal deviates by {diff_pct:+.1f}% from step median ({median_sig:.1f} RU). "
                                f"Likely micro-bubble or probe contact artifact."
                            )

        # 2. Tag-level pipetting outlier check (for rows not already flagged as bubble)
        if schema.TAG in work.columns:
            for tag_val, group in work.dropna(subset=[schema.TAG]).groupby(schema.TAG):
                if len(group) >= self.min_replicates:
                    signals = group[schema.SIGNAL_RU].to_numpy(dtype=float)
                    median_sig = float(np.nanmedian(signals))
                    q25, q75 = np.nanpercentile(signals, [25, 75])
                    iqr = max(q75 - q25, abs(median_sig) * 0.04)
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr

                    for idx in group.index:
                        if anomaly_types[idx] == AnomalyType.NORMAL:
                            sig = work.at[idx, schema.SIGNAL_RU]
                            if pd.notna(sig) and (sig < lower_bound or sig > upper_bound):
                                anomaly_types[idx] = AnomalyType.PIPETTING_OUTLIER
                                score = abs(sig - float(np.nanmedian(signals))) / max(iqr, 1.0)
                                anomaly_scores[idx] = float(score)
                                recommendations[idx] = (
                                    f"Measurement lies outside tag replicate IQR [{lower_bound:.1f}, {upper_bound:.1f}]. "
                                    f"Possible sample volume or pipetting variation."
                                )

        # 3. Compensation factor tolerance check
        for idx in range(n_rows):
            comp = work.at[idx, schema.SIGNAL_COMPENSATION]
            if pd.notna(comp) and abs(float(comp) - 1.0) > self.compensation_tol:
                if anomaly_types[idx] == AnomalyType.NORMAL:
                    anomaly_types[idx] = AnomalyType.COMPENSATION_ANOMALY
                    anomaly_scores[idx] = float(abs(float(comp) - 1.0) / self.compensation_tol)
                    recommendations[idx] = (
                        f"Signal compensation factor ({comp:.3f}) exceeds nominal tolerance (±{self.compensation_tol*100:.0f}%). "
                        f"Verify sensor strip calibration and baseline."
                    )

        # Build output dataframe
        anomalies_df = pd.DataFrame(
            {
                schema.STEP: work[schema.STEP],
                schema.TAG: work.get(schema.TAG, pd.Series(index=work.index, dtype=object)),
                schema.SIGNAL_RU: work[schema.SIGNAL_RU],
                "anomaly_type": [a.value for a in anomaly_types],
                "anomaly_score": anomaly_scores,
                "recommendation": recommendations,
                "is_anomaly": [a != AnomalyType.NORMAL for a in anomaly_types],
            }
        )

        is_anom = np.array([a != AnomalyType.NORMAL for a in anomaly_types])
        cleaned = work.loc[~is_anom].copy().reset_index(drop=True)

        return AnomalyReport(
            anomalies=anomalies_df,
            num_anomalies=int(np.sum(is_anom)),
            cleaned_df=cleaned,
        )


def detect_sensor_anomalies(
    df: pd.DataFrame,
    *,
    replicate_z_thresh: float = 2.5,
    compensation_tol: float = 0.05,
) -> AnomalyReport:
    """Convenience function to run sensor anomaly detection on an Amperia export DataFrame."""
    detector = SensorAnomalyDetector(
        replicate_z_thresh=replicate_z_thresh,
        compensation_tol=compensation_tol,
    )
    return detector.detect(df)
