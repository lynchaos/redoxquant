"""Quickstart: parse a demo export, refit the curve, quantify and QC.

Run from the repo root:  python examples/quickstart.py
"""

import sys
from pathlib import Path

# Allow running directly from repo root without prior editable install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

import redoxquant as rq
from redoxquant import schema

DEMO = Path(__file__).parent.parent / "tests" / "data" / "demo_tocilizumab.csv"


def main() -> None:
    df = rq.read_csv(DEMO)
    print(f"Parsed {len(df)} measurements\n")

    # In a real export the standard rows define the curve. Here we refit from
    # the demo's (concentration, signal) anchor points to illustrate the flow.
    anchors_x = np.append(df[schema.CONCENTRATION].to_numpy(), [9.45, 77.1])
    anchors_y = np.append(df[schema.SIGNAL_RU].to_numpy(), [2547.8, 807.8])
    curve = rq.fit_calibration(anchors_x, anchors_y, model="5PL")
    print(f"Fitted {curve.model}: descending={curve.descending}, R^2={curve.r_squared:.4f}")
    print(f"  a={curve.a:.1f} b={curve.b:.2f} c={curve.c:.2f} d={curve.d:.1f} g={curve.g:.2f}\n")

    q = rq.quantify(df, curve)
    cols = [schema.SIGNAL_RU, schema.CONCENTRATION, "concentration_calc",
            schema.ADJUSTED_CONCENTRATION, "adjusted_concentration_calc", schema.TAG]
    print(q[cols].round(2).to_string(index=False))

    print("\nPer-tag summary:")
    print(rq.group_stats(df).round(2).to_string(index=False))

    # Demonstrate Delta-Method 95% Confidence Interval on back-calculation
    sample_signal = df[schema.SIGNAL_RU].iloc[0]
    ci = curve.back_calculate_with_ci(sample_signal, ci=0.95)
    print(f"\nSample 1 Back-Calc 95% CI: {ci.estimate:.2f} µg/ml [{ci.lower:.2f}, {ci.upper:.2f}]")

    # Demonstrate Standalone HTML Report Generation
    report_path = Path(__file__).parent / "demo_report.html"
    rq.generate_html_report(df, curve, title="Tocilizumab Quantification Run", output_path=report_path)
    print(f"\nGenerated interactive run report: {report_path}")


if __name__ == "__main__":
    main()
