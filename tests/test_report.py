"""Tests for standalone HTML report generation."""

import numpy as np

from redoxquant import fit_calibration, generate_html_report, read_csv, schema


def test_generate_html_report(demo_csv_path, anchor_points, tmp_path):
    df = read_csv(demo_csv_path)
    x = np.array([p[0] for p in anchor_points])
    y = np.array([p[1] for p in anchor_points])
    curve = fit_calibration(x, y, model="5PL", weight="1/y2")

    out_file = tmp_path / "test_report.html"
    html_str = generate_html_report(df, curve, title="Test Tocilizumab Report", output_path=out_file)

    assert "<!DOCTYPE html>" in html_str
    assert "Test Tocilizumab Report" in html_str
    assert "<svg" in html_str
    assert "5PL" in html_str
    assert out_file.exists()
    assert len(out_file.read_text(encoding="utf-8")) > 500
