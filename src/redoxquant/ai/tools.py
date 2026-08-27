"""OpenAI / Qwen Function Calling Tool Specifications for redoxquant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .. import bioanalysis, curve as _curve, io, ml, report, schema
from ..quantify import group_stats as _group_stats, qc_flags as _qc_flags, quantify as _quantify


TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "load_amperia_file",
            "description": "Load an Amperia CSV or XLSX export file into a canonical dataframe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the CSV or XLSX file."}
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fit_standard_curve",
            "description": "Fit a 5PL or 4PL calibration curve from standard concentration and signal points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "enum": ["5PL", "4PL"], "default": "5PL"},
                    "weight": {"type": "string", "enum": ["1/y2", "1/y", "None"], "default": "1/y2"},
                    "use_bayesian": {"type": "boolean", "default": False, "description": "Use Bayesian prior for few-shot standards"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "quantify_dataset",
            "description": "Quantify samples using the fitted standard curve, computing replicate statistics and QC flags.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_sensor_anomalies",
            "description": "Scan dataset for microfluidic bubbles, electrode fouling, and pipetting outliers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "replicate_z_thresh": {"type": "number", "default": 2.5},
                    "compensation_tol": {"type": "number", "default": 0.05},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_assay_limits",
            "description": "Calculate LOD (Limit of Detection), LLOQ, and ULOQ from the standard curve.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_dilution_linearity",
            "description": "Evaluate sample dilutional linearity and parallelism across multiple dilution factors.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_html_report",
            "description": "Generate a standalone publication-ready HTML run report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_path": {"type": "string", "description": "Destination file path for the HTML report."}
                },
                "required": ["output_path"],
            },
        },
    },
]


class ToolExecutor:
    """Stateful runtime for executing redoxquant tool calls from AI agents."""

    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.curve: Optional[Any] = None

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call and return a JSON string summary."""
        try:
            if tool_name == "load_amperia_file":
                fp = Path(arguments["filepath"])
                if fp.suffix.lower() == ".xlsx":
                    self.df = io.read_xlsx(fp)
                else:
                    self.df = io.read_csv(fp)
                return json.dumps({
                    "status": "success",
                    "rows_loaded": len(self.df),
                    "tags": list(self.df[schema.TAG].dropna().unique()),
                    "columns": list(self.df.columns),
                })

            elif tool_name == "fit_standard_curve":
                if self.df is None:
                    return json.dumps({"status": "error", "message": "No data loaded. Call load_amperia_file first."})
                model = arguments.get("model", "5PL")
                weight = arguments.get("weight", "1/y2")
                use_bayesian = arguments.get("use_bayesian", False)

                # Filter standard rows or anchor points
                std_rows = self.df[self.df[schema.SOLUTION_TYPE] == schema.SolutionType.STANDARD.value]
                if len(std_rows) < 3:
                    std_rows = self.df.dropna(subset=[schema.CONCENTRATION, schema.SIGNAL_RU])

                x = std_rows[schema.CONCENTRATION].to_numpy(dtype=float)
                y = std_rows[schema.SIGNAL_RU].to_numpy(dtype=float)

                if use_bayesian:
                    self.curve = ml.fit_bayesian_5pl(x, y, weight=weight)
                else:
                    self.curve = _curve.fit_calibration(x, y, model=model, weight=weight)

                return json.dumps({
                    "status": "success",
                    "model": str(self.curve.model) if hasattr(self.curve, "model") else "Bayesian-5PL",
                    "descending": bool(self.curve.descending),
                    "r_squared": float(round(self.curve.r_squared, 4)),
                    "parameters": {
                        "a": float(round(self.curve.a, 2)),
                        "b": float(round(self.curve.b, 2)),
                        "c": float(round(self.curve.c, 2)),
                        "d": float(round(self.curve.d, 2)),
                        "g": float(round(self.curve.g, 2)),
                    },
                })

            elif tool_name == "quantify_dataset":
                if self.df is None or self.curve is None:
                    return json.dumps({"status": "error", "message": "Need both loaded data and fitted curve."})
                q_df = _quantify(self.df, self.curve)
                flags = _qc_flags(q_df)
                stats = _group_stats(q_df)
                return json.dumps({
                    "status": "success",
                    "group_stats": stats.to_dict(orient="records"),
                    "compensation_flags": int(flags["flag_compensation"].sum()),
                    "high_cv_flags": int(flags["flag_high_cv"].sum()),
                })


            elif tool_name == "detect_sensor_anomalies":
                if self.df is None:
                    return json.dumps({"status": "error", "message": "No data loaded."})
                report_obj = ml.detect_sensor_anomalies(
                    self.df,
                    replicate_z_thresh=float(arguments.get("replicate_z_thresh", 2.5)),
                    compensation_tol=float(arguments.get("compensation_tol", 0.05)),
                )
                anom_rows = report_obj.anomalies[report_obj.anomalies["is_anomaly"]]
                return json.dumps({
                    "status": "success",
                    "num_anomalies": int(report_obj.num_anomalies),
                    "anomalies": anom_rows.to_dict(orient="records"),
                })

            elif tool_name == "check_assay_limits":
                if self.df is None or self.curve is None:
                    return json.dumps({"status": "error", "message": "Need both loaded data and fitted curve."})
                limits = bioanalysis.compute_assay_limits(self.df, self.curve)
                return json.dumps({
                    "status": "success",
                    "lod": float(round(limits.lod, 3)) if np.isfinite(limits.lod) else None,
                    "lloq": float(round(limits.lloq, 3)) if np.isfinite(limits.lloq) else None,
                    "uloq": float(round(limits.uloq, 3)) if np.isfinite(limits.uloq) else None,
                })

            elif tool_name == "check_dilution_linearity":
                if self.df is None or self.curve is None:
                    return json.dumps({"status": "error", "message": "Need both loaded data and fitted curve."})
                lin = bioanalysis.evaluate_dilution_linearity(self.df, self.curve)
                return json.dumps({
                    "status": "success",
                    "is_linear": bool(lin.is_linear),
                    "summary": lin.summary.to_dict(orient="records"),
                })

            elif tool_name == "generate_html_report":
                if self.df is None or self.curve is None:
                    return json.dumps({"status": "error", "message": "Need both loaded data and fitted curve."})
                out_p = arguments["output_path"]
                report.generate_html_report(self.df, self.curve, output_path=out_p)
                return json.dumps({"status": "success", "report_path": str(out_p)})

            else:
                return json.dumps({"status": "error", "message": f"Unknown tool: {tool_name}"})

        except Exception as e:
            return json.dumps({"status": "error", "error_type": type(e).__name__, "message": str(e)})
