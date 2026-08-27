"""Tests for ML sensor anomaly detection module."""

import numpy as np
import pandas as pd

from redoxquant import schema
from redoxquant.ml.anomaly import AnomalyType, SensorAnomalyDetector, detect_sensor_anomalies
from redoxquant.synthetic import generate_standards


def test_anomaly_detection_clean_data():
    std = generate_standards(seed=42, cv=0.01, replicates=4)
    report = detect_sensor_anomalies(std)

    assert not report.has_anomalies
    assert report.num_anomalies == 0
    assert len(report.cleaned_df) == len(std)


def test_detects_bubble_spike():
    std = generate_standards(seed=42, cv=0.01, replicates=4)
    df = std.copy()
    # Introduce a massive bubble artifact in row 2 (step 1)
    df.at[2, schema.SIGNAL_RU] = df.at[2, schema.SIGNAL_RU] * 3.5

    detector = SensorAnomalyDetector(replicate_z_thresh=2.5)
    report = detector.detect(df)

    assert report.has_anomalies
    assert report.num_anomalies == 1
    assert report.anomalies.at[2, "anomaly_type"] == AnomalyType.BUBBLE_ARTIFACT.value
    assert report.anomalies.at[2, "is_anomaly"]
    assert len(report.cleaned_df) == len(df) - 1


def test_detects_compensation_anomaly():
    std = generate_standards(seed=42, cv=0.01, replicates=4)
    df = std.copy()
    # Introduce bad compensation factor
    df.at[5, schema.SIGNAL_COMPENSATION] = 1.15  # 15% shift

    report = detect_sensor_anomalies(df, compensation_tol=0.05)
    assert report.has_anomalies
    assert report.anomalies.at[5, "anomaly_type"] == AnomalyType.COMPENSATION_ANOMALY.value
