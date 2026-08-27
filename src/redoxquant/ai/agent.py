"""Qwen AI Bioprocess Copilot and Autonomous Diagnostic Engine for redoxquant."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .tools import TOOL_DEFINITIONS, ToolExecutor

BIOPROCESS_SYSTEM_PROMPT = """You are the RedoxQuant AI Copilot — an expert bioprocess bioanalytical scientist and data automation assistant.
You assist bench scientists, bioprocess engineers, and analytical chemists in analyzing protein quantification data from Amperia (Redox Electrochemical Detection) exports.

Your responsibilities:
1. Orchestrate redoxquant tools to load data, fit 5PL standard curves, evaluate replicate precision (%CV), and quantify sample concentrations.
2. Analyze Quality Control (QC) flags: explain why assays fail (e.g. signal compensation shifts > 5%, high %CV > 20%, or dilutional non-linearity).
3. Diagnose electrochemical sensor issues: distinguish between micro-bubble artifacts on the sensor strip, electrode fouling, and pipetting volume errors.
4. Provide clear, rigorous, GLP-formatted summaries of assay runs with actionable recommendations.
"""


class DiagnosticReasoner:
    """Offline rule-based bioanalytical diagnostic engine for automated assay troubleshooting."""

    @staticmethod
    def diagnose_run(executor: ToolExecutor) -> Dict[str, Any]:
        """Perform a full diagnostic audit on the currently loaded run in ToolExecutor."""
        if executor.df is None:
            return {
                "summary": "No dataset loaded.",
                "findings": ["Dataset is empty. Call load_amperia_file first."],
                "recommendations": ["Load an Amperia export CSV or XLSX file to begin analysis."],
            }

        findings = []
        recommendations = []

        # 1. Anomaly check
        anom_res = json.loads(executor.execute("detect_sensor_anomalies", {}))
        num_anom = anom_res.get("num_anomalies", 0)
        if num_anom > 0:
            findings.append(f"Detected {num_anom} sensor probe anomal{'y' if num_anom==1 else 'ies'}.")
            for anom in anom_res.get("anomalies", []):
                recommendations.append(f"Step {anom.get('step')}: {anom.get('recommendation')}")
        else:
            findings.append("No microfluidic bubble or electrode fouling artifacts detected.")

        # 2. Curve fit quality check
        if executor.curve is not None:
            r2 = executor.curve.r_squared
            if r2 < 0.98:
                findings.append(f"Warning: Standard curve R² ({r2:.4f}) is below standard acceptance threshold (0.98).")
                recommendations.append("Review standard anchor points for potential outlier wells.")
            else:
                findings.append(f"Standard curve fit is excellent (R² = {r2:.4f}, model = 5PL).")

        # 3. Quantification & QC checks
        if executor.df is not None and executor.curve is not None:
            q_res = json.loads(executor.execute("quantify_dataset", {}))
            comp_flags = q_res.get("compensation_flags", 0)
            high_cv_flags = q_res.get("high_cv_flags", 0)

            if comp_flags > 0:
                findings.append(f"{comp_flags} measurement(s) exceeded signal compensation tolerance (±5%).")
                recommendations.append("Check sensor strip baseline calibration.")
            if high_cv_flags > 0:
                findings.append(f"{high_cv_flags} measurement group(s) exceeded 20% replicate %CV.")
                recommendations.append("Investigate sample mixing / pipetting uniformity.")
            if comp_flags == 0 and high_cv_flags == 0:
                findings.append("All sample replicate groups passed QC precision and compensation gates.")

        return {
            "summary": "Assay run passed all QC gates." if not recommendations else "Assay run flagged items for review.",
            "findings": findings,
            "recommendations": recommendations,
        }


class RedoxCopilot:
    """Conversational AI Agent compatible with Qwen2.5 (via Ollama/vLLM/OpenAI-compatible endpoints)."""

    def __init__(
        self,
        *,
        model: str = "qwen2.5-coder:7b",
        api_base: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.executor = ToolExecutor()
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": BIOPROCESS_SYSTEM_PROMPT}
        ]

    def chat(self, user_message: str, max_turns: int = 6) -> str:
        """Process a user query with multi-turn tool calling against local or remote Qwen endpoint."""
        try:
            import httpx
        except ImportError:
            # Fallback to diagnostic reasoner if httpx is not installed
            diag = DiagnosticReasoner.diagnose_run(self.executor)
            return (
                f"Note: Optional dependency `httpx` not installed for live Qwen API.\n\n"
                f"Automated Offline Diagnostics:\n"
                f"• {diag['summary']}\n"
                + "\n".join(f"  - {f}" for f in diag["findings"])
                + ("\nRecommendations:\n" + "\n".join(f"  - {r}" for r in diag["recommendations"]) if diag["recommendations"] else "")
            )

        self.messages.append({"role": "user", "content": user_message})

        for _ in range(max_turns):
            payload = {
                "model": self.model,
                "messages": self.messages,
                "tools": TOOL_DEFINITIONS,
                "tool_choice": "auto",
            }

            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(
                        f"{self.api_base}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
            except Exception as err:
                # If no local Qwen server is running, provide graceful diagnostic fallback
                diag = DiagnosticReasoner.diagnose_run(self.executor)
                return (
                    f"(Could not connect to Qwen server at {self.api_base}: {err})\n\n"
                    f"Automated Offline Diagnostic Report:\n"
                    f"Status: {diag['summary']}\n"
                    + "\n".join(f"• {f}" for f in diag["findings"])
                )

            msg = data["choices"][0]["message"]
            self.messages.append(msg)

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                return str(msg.get("content", ""))

            # Execute tool calls
            for tc in tool_calls:
                fn = tc["function"]
                fn_name = fn["name"]
                fn_args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]

                result_str = self.executor.execute(fn_name, fn_args)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": fn_name,
                    "content": result_str,
                })

        return "Reached maximum tool execution turns."
