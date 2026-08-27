"""Tests for Qwen AI Copilot tool specifications and execution runtime."""

import json
from pathlib import Path

from redoxquant.ai.agent import DiagnosticReasoner, RedoxCopilot
from redoxquant.ai.tools import TOOL_DEFINITIONS, ToolExecutor
from redoxquant.synthetic import generate_standards


def test_tool_definitions_schema():
    assert len(TOOL_DEFINITIONS) >= 6
    names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
    assert "load_amperia_file" in names
    assert "fit_standard_curve" in names
    assert "quantify_dataset" in names
    assert "detect_sensor_anomalies" in names
    assert "check_assay_limits" in names
    assert "generate_html_report" in names


def test_tool_executor_pipeline(tmp_path):
    # Create synthetic export CSV with standards
    std_df = generate_standards(seed=42, cv=0.01, replicates=4)
    std_file = tmp_path / "standards_export.csv"
    std_df.to_csv(std_file, index=False)

    executor = ToolExecutor()

    # 1. Load data
    res1 = json.loads(executor.execute("load_amperia_file", {"filepath": str(std_file)}))
    assert res1["status"] == "success"
    assert res1["rows_loaded"] == 32

    # 2. Fit curve
    res2 = json.loads(executor.execute("fit_standard_curve", {"model": "5PL"}))
    assert res2["status"] == "success"
    assert res2["r_squared"] > 0.98

    # 3. Detect anomalies
    res3 = json.loads(executor.execute("detect_sensor_anomalies", {}))
    assert res3["status"] == "success"
    assert res3["num_anomalies"] == 0

    # 4. Assay limits
    res_lim = json.loads(executor.execute("check_assay_limits", {}))
    assert res_lim["status"] == "success"
    assert res_lim["lloq"] is not None

    # 5. Diagnostic reasoner audit
    diag = DiagnosticReasoner.diagnose_run(executor)
    assert "summary" in diag
    assert len(diag["findings"]) > 0

    # 6. Report generation
    out_report = tmp_path / "copilot_report.html"
    res6 = json.loads(executor.execute("generate_html_report", {"output_path": str(out_report)}))
    assert res6["status"] == "success"
    assert out_report.exists()


def test_copilot_offline_chat():
    copilot = RedoxCopilot()
    # Offline chat fallback should produce clean diagnostic summary without crashing
    response = copilot.chat("Check this assay run for anomalies.")
    assert "Diagnostic" in response or "summary" in response.lower()
