"""Standalone interactive HTML and visual report generator for Amperia exports."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from . import schema
from .curve import CalibrationCurve
from .quantify import group_stats, qc_flags, quantify


def _render_svg_curve(curve: CalibrationCurve, df: pd.DataFrame, width: int = 680, height: int = 340) -> str:
    """Render a clean SVG plot of the 5PL standard curve and sample points."""
    signals = df[schema.SIGNAL_RU].dropna().to_numpy(dtype=float)
    concs = df[schema.CONCENTRATION].dropna().to_numpy(dtype=float)

    # Determine x (concentration) and y (signal) limits
    min_x = max(float(np.min(concs)) * 0.5, 0.1) if len(concs) > 0 else 0.1
    max_x = max(float(np.max(concs)) * 2.0, 100.0) if len(concs) > 0 else 100.0

    min_y = min(float(np.min(signals)) * 0.9, curve.d * 0.9) if len(signals) > 0 else min(curve.a, curve.d) * 0.9
    max_y = max(float(np.max(signals)) * 1.1, curve.a * 1.1) if len(signals) > 0 else max(curve.a, curve.d) * 1.1

    pad_l, pad_r, pad_t, pad_b = 60, 30, 20, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    # Log10 scale for concentration (X)
    log_min_x = np.log10(min_x)
    log_max_x = np.log10(max_x)

    def map_x(x_val: float) -> float:
        if x_val <= 0:
            return pad_l
        lx = np.log10(x_val)
        norm = (lx - log_min_x) / (log_max_x - log_min_x + 1e-12)
        return pad_l + norm * plot_w

    def map_y(y_val: float) -> float:
        norm = (y_val - min_y) / (max_y - min_y + 1e-12)
        return pad_t + plot_h - norm * plot_h

    # Curve points
    curve_x = np.geomspace(min_x, max_x, 100)
    curve_y = curve.predict(curve_x)
    points_svg = " ".join(f"{map_x(x):.1f},{map_y(y):.1f}" for x, y in zip(curve_x, curve_y) if np.isfinite(y))

    # Grid & Axes
    svg_elements = []
    # Background
    svg_elements.append(f'<rect width="{width}" height="{height}" fill="#f8fafc" rx="8" />')
    svg_elements.append(f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#e2e8f0" />')

    # Fitted Curve Line
    svg_elements.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{points_svg}" />')

    # Data Points
    for _, row in df.iterrows():
        c_val = row.get(schema.CONCENTRATION)
        s_val = row.get(schema.SIGNAL_RU)
        if pd.notna(c_val) and pd.notna(s_val) and c_val > 0:
            px = map_x(float(c_val))
            py = map_y(float(s_val))
            tag = str(row.get(schema.TAG, "Sample"))
            svg_elements.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="#ef4444" stroke="#ffffff" stroke-width="1.5">'
                f'<title>{tag}: {c_val} µg/ml, {s_val} RU</title></circle>'
            )

    # Axis Labels
    svg_elements.append(f'<text x="{pad_l + plot_w/2}" y="{height - 8}" text-anchor="middle" font-size="12" fill="#475569" font-family="system-ui, sans-serif">Concentration (log scale)</text>')
    svg_elements.append(f'<text transform="rotate(-90)" x="-{pad_t + plot_h/2}" y="18" text-anchor="middle" font-size="12" fill="#475569" font-family="system-ui, sans-serif">Signal (RU)</text>')

    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">{"".join(svg_elements)}</svg>'


def generate_html_report(
    df: pd.DataFrame,
    curve: CalibrationCurve,
    *,
    title: str = "Amperia Quantification Report",
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate a clean, standalone HTML run report for an Amperia export.

    Parameters
    ----------
    df:
        Canonical measurements DataFrame.
    curve:
        Fitted calibration curve.
    title:
        Report header title.
    output_path:
        Optional file path to write the HTML output.

    Returns
    -------
    str
        Complete HTML document string.
    """
    quant_df = quantify(df, curve)
    flagged_df = qc_flags(quant_df)
    stats_df = group_stats(quant_df)

    num_total = len(flagged_df)
    comp_flags = flagged_df["flag_compensation"].fillna(False).astype(bool)
    cv_flags = flagged_df["flag_high_cv"].fillna(False).astype(bool)
    num_flagged = int(comp_flags.sum() + cv_flags.sum())
    orientation = "Descending (Competitive)" if curve.descending else "Ascending (Sandwich)"

    svg_chart = _render_svg_curve(curve, df)

    # Build Sample Rows
    rows_html = []
    for _, r in flagged_df.iterrows():
        step = r[schema.STEP]
        tag_val = r.get(schema.TAG)
        tag = str(tag_val) if pd.notna(tag_val) else "—"
        sig = f"{r[schema.SIGNAL_RU]:.1f}" if pd.notna(r[schema.SIGNAL_RU]) else "—"
        comp = f"{r[schema.SIGNAL_COMPENSATION]:.3f}" if pd.notna(r[schema.SIGNAL_COMPENSATION]) else "1.000"
        rep_conc = f"{r[schema.CONCENTRATION]:.2f}" if pd.notna(r[schema.CONCENTRATION]) else "—"
        calc_conc = f"{r['concentration_calc']:.2f}" if pd.notna(r.get('concentration_calc')) else "—"
        adj_conc = f"{r['adjusted_concentration_calc']:.2f}" if pd.notna(r.get('adjusted_concentration_calc')) else "—"
        dil = f"{r[schema.DILUTION_FACTOR]:g}" if pd.notna(r[schema.DILUTION_FACTOR]) else "1"

        flags = []
        if bool(pd.notna(r.get("flag_compensation")) and r["flag_compensation"]):
            flags.append('<span class="badge badge-warn">Comp Warn</span>')
        if bool(pd.notna(r.get("flag_high_cv")) and r["flag_high_cv"]):
            flags.append('<span class="badge badge-error">High CV</span>')
        if not flags:
            flags.append('<span class="badge badge-ok">Pass</span>')
        flags_html = " ".join(flags)

        rows_html.append(
            f"<tr>"
            f"<td>{step}</td><td><strong>{html.escape(str(tag))}</strong></td><td>{sig}</td>"
            f"<td>{comp}</td><td>{dil}</td><td>{rep_conc}</td><td>{calc_conc}</td>"
            f"<td><strong>{adj_conc}</strong></td><td>{flags_html}</td>"
            f"</tr>"
        )


    # Build Stats Rows
    stats_html = []
    for _, s in stats_df.iterrows():
        t_val = s.get(schema.TAG)
        t = str(t_val) if pd.notna(t_val) else "—"
        n = s["n"]
        mean_sig = f"{s['mean']:.1f}" if pd.notna(s.get('mean')) else "—"
        std_sig = f"{s['std_dev']:.2f}" if pd.notna(s.get('std_dev')) else "—"
        cv = f"{s['cv_pct']:.2f}%" if pd.notna(s.get('cv_pct')) else "—"
        mean_c = f"{s['mean_concentration']:.2f}" if "mean_concentration" in s and pd.notna(s["mean_concentration"]) else "—"
        stats_html.append(
            f"<tr><td><strong>{html.escape(str(t))}</strong></td><td>{n}</td><td>{mean_sig}</td>"
            f"<td>{std_sig}</td><td>{cv}</td><td>{mean_c}</td></tr>"
        )


    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 24px; background: #f1f5f9; color: #1e293b; }}
  .container {{ max-width: 1040px; margin: 0 auto; background: #ffffff; border-radius: 12px; padding: 32px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }}
  h1 {{ font-size: 24px; margin-top: 0; color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
  h2 {{ font-size: 18px; margin-top: 28px; color: #334155; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 20px 0; }}
  .metric-card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }}
  .metric-label {{ font-size: 12px; text-transform: uppercase; color: #64748b; font-weight: 600; }}
  .metric-value {{ font-size: 20px; font-weight: 700; color: #0f172a; margin-top: 4px; }}
  .chart-container {{ text-align: center; margin: 24px 0; background: #f8fafc; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f8fafc; color: #475569; font-weight: 600; font-size: 13px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
  .badge-ok {{ background: #dcfce7; color: #166534; }}
  .badge-warn {{ background: #fef9c3; color: #854d0e; }}
  .badge-error {{ background: #fee2e2; color: #991b1b; }}
  .footer {{ margin-top: 36px; padding-top: 16px; border-top: 1px solid #e2e8f0; font-size: 12px; color: #94a3b8; text-align: center; }}
</style>
</head>
<body>
<div class="container">
  <h1>{html.escape(title)}</h1>
  
  <div class="metrics-grid">
    <div class="metric-card">
      <div class="metric-label">Model & Orientation</div>
      <div class="metric-value">{curve.model} ({orientation})</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Curve Fit R²</div>
      <div class="metric-value">{curve.r_squared:.4f}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Measurements</div>
      <div class="metric-value">{num_total}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">QC Alerts</div>
      <div class="metric-value">{num_flagged}</div>
    </div>
  </div>

  <h2>Standard Curve Fit (5PL Parameters)</h2>
  <div class="metric-card" style="font-family: monospace; font-size: 13px;">
    a (x→0) = {curve.a:.2f} | b (Hill) = {curve.b:.2f} | c (EC50) = {curve.c:.2f} | d (x→∞) = {curve.d:.2f} | g (Asym) = {curve.g:.2f}
  </div>

  <div class="chart-container">
    {svg_chart}
  </div>

  <h2>Sample Replicate Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Tag</th>
        <th>Replicates (n)</th>
        <th>Mean Signal (RU)</th>
        <th>Signal Std Dev</th>
        <th>Signal %CV</th>
        <th>Mean Conc</th>
      </tr>
    </thead>
    <tbody>
      {''.join(stats_html)}
    </tbody>
  </table>

  <h2>Detailed Measurements</h2>
  <table>
    <thead>
      <tr>
        <th>Step</th>
        <th>Tag</th>
        <th>Signal (RU)</th>
        <th>Comp Factor</th>
        <th>Dilution</th>
        <th>Reported Conc</th>
        <th>Calc Conc</th>
        <th>Adjusted Conc</th>
        <th>QC Status</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>

  <div class="footer">
    Generated with redoxquant (Research Use Only companion for Amperia analysis exports).
  </div>
</div>
</body>
</html>
"""
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html_content, encoding="utf-8")

    return html_content
