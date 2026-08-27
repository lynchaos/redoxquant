<p align="center">
  <img src="redoxquant_logo.png" width="220" alt="RedoxQuant Logo" />
</p>

<h1 align="center">redoxquant</h1>

<p align="center">
  <strong>Scientific Machine Learning & Automated Bioanalytical Intelligence for Redox Electrochemical Assays</strong>
</p>

<p align="center">
  <a href="https://github.com/lynchaos/redoxquant/actions/workflows/ci.yml"><img src="https://github.com/lynchaos/redoxquant/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://pypi.org/project/redoxquant/"><img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT" /></a>
  <a href="https://kemal.yaylali.uk"><img src="https://img.shields.io/badge/author-Kemal%20Yaylalı-purple" alt="Author" /></a>
  <a href="mailto:support@yaylali.uk"><img src="https://img.shields.io/badge/support-support%40yaylali.uk-orange" alt="Support Email" /></a>
</p>


---

## Vision & Philosophy

**`redoxquant`** is an open scientific computing and machine learning companion engineered for **Amperia™** (Redox Electrochemical Detection) protein-quantification analysis exports.

Benchtop bio-instruments often suffer from the *"touchscreen island"* problem: high-precision electrochemical readings get trapped inside standalone embedded GUIs and static USB export files. As modern bioprocess engineering moves toward continuous manufacturing, digital twins, and autonomous Process Analytical Technology (PAT), we need a **rigorous mathematical and machine learning substrate** to model, audit, and bridge electrochemical biosensor telemetry into downstream computational pipelines.

Created from a deep curiosity at the intersection of **electrochemical biophysics**, **non-linear statistical calibration**, and **modern machine learning**, `redoxquant` transforms raw Response Units (RU) into verifiable, auditable, uncertainty-aware bioanalytical insights.

> **Disclaimer**: *Not affiliated with or endorsed by Abselion / HexagonFab Ltd. "Amperia" and "Abselion" are trademarks of their owner, used solely to describe file format compatibility. `redoxquant` operates strictly downstream of the instrument's exported files and does not touch proprietary firmware. Research Use Only (RUO).*

---

## Key Capabilities

```
                       ┌─────────────────────────────────────────────────────────┐
                       │               Amperia CSV / XLSX Export                 │
                       └────────────────────────────┬────────────────────────────┘
                                                    │
                                                    ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │          redoxquant Canonical Ingestion Layer           │
                       └──────────────┬───────────────────────────┬──────────────┘
                                      │                           │
                   ┌──────────────────┴─────────────┐             │
                   ▼                                ▼             ▼
     ┌───────────────────────────┐    ┌───────────────────────────────┐
     │  ml.anomaly               │    │  curve / ml.bayesian          │
     │  • Micro-bubble isolation │    │  • 5PL / 4PL Curve Fitting    │
     │  • Fouling detection      │    │  • Delta-Method 95% CIs       │
     │  • Robust MAD z-scores    │    │  • Few-Shot MAP Prior Fitting │
     └─────────────┬─────────────┘    └───────────────┬───────────────┘
                   │                                  │
                   └──────────────────┬───────────────┘
                                      │
                                      ▼
     ┌───────────────────────────────────────────────────────────────────────────┐
     │  bioanalysis & comparison                                                 │
     │  • LOD / LLOQ / ULOQ Dynamic Range Estimation                             │
     │  • Total Error Profiles (%TE = |%RE| + 2·%CV per Azadeh et al. 2018)      │
     │  • Dilutional Linearity & Parallelism Checks                              │
     │  • Orthogonal Method Regression (Deming, Passing–Bablok, Bland–Altman)    │
     └────────────────────────────────┬──────────────────────────────────────────┘
                                      │
                   ┌──────────────────┴────────────────┐
                   ▼                                   ▼
     ┌───────────────────────────┐       ┌───────────────────────────┐
     │  ml.surrogate             │       │  ai.agent                 │
     │  • Cross-Method Bridge    │       │  • Qwen2.5 Copilot        │
     │  • Amperia ↔ ELISA/HPLC   │       │  • Tool-Calling Runtime   │
     │  • Epistemic Bands        │       │  • Diagnostic Reasoner    │
     └───────────────────────────┘       └─────────────┬─────────────┘
                                                       │
                                                       ▼
                                         ┌───────────────────────────┐
                                         │  report.generate_html     │
                                         │  • Embedded SVG 5PL Curve │
                                         │  • GLP Audit Summary      │
                                         └───────────────────────────┘
```

### 1. Biophysical 5PL & 4PL Calibration
- Implements the 5-parameter logistic sigmoidal equation ([Gottschalk & Dunn, 2005](https://doi.org/10.1016/j.ab.2005.04.035)):
  $$y = d + \frac{a - d}{\left(1 + \left(\frac{x}{c}\right)^b\right)^g}$$
- Variance-stabilizing $1/y^2$ weighting ([Azadeh et al., 2018](https://doi.org/10.1208/s12248-017-0159-4)) handling heteroscedastic noise.
- Automatic detection of ascending (sandwich) and descending (competitive / inhibition) assay orientations.

### 2. Analytical & Bayesian Uncertainty Estimation
- **First-Order Delta Method**: Propagates parameter covariance ($\mathbf{\Sigma}_{\boldsymbol{\theta}}$) through the numerical Jacobian of the 5PL inverse to construct exact 95% Confidence Intervals for back-calculated concentrations.
- **Few-Shot Bayesian Calibration** (`redoxquant.ml.bayesian`): Maximum A Posteriori (MAP) estimation conditioned on empirical historical priors, enabling accurate 5PL fitting with as few as **2–3 calibration points** ($>80\%$ reduction in standard reagents) with full posterior sampling.

### 3. Bioanalytical Quality Control & Dynamic Range
- Automated **LOD**, **LLOQ**, and **ULOQ** determination.
- **Total Error Profiling**: Computes $\text{TE} = |\%RE| + 2 \cdot \%CV$ across standard levels to guarantee bioanalytical compliance ($<30\%$ in-range, $<40\%$ at LLOQ/ULOQ).
- **Dilutional Linearity**: Parallelism testing across multi-level sample dilution series to identify matrix effects and non-specific binding.

### 4. Orthogonal Method Comparison (CLSI EP09-A3)
- **Deming Regression**: Orthogonal linear regression accounting for analytical error in both test and reference methods ([Linnet, 1993](https://pubmed.ncbi.nlm.nih.gov/8448852/)).
- **Passing–Bablok Non-Parametric Regression**: Robust rank-based slope/intercept estimation invariant to outliers ([Passing & Bablok, 1983](https://doi.org/10.1515/cclm.1983.21.11.709)).
- **Bland–Altman Agreement**: Mean bias and 95% Limits of Agreement ([Bland & Altman, 1986](https://doi.org/10.1016/S0140-6736(86)90837-8)).

### 5. Sensor Anomaly & Fault Classifier (`redoxquant.ml.anomaly`)
- Unsupervised multi-probe replicate analysis using Median Absolute Deviation (MAD) and robust z-scoring.
- Automatically flags and classifies `BUBBLE_ARTIFACT`, `ELECTRODE_FOULING`, `PIPETTING_OUTLIER`, and `COMPENSATION_ANOMALY` before standard curve fitting.

### 6. Cross-Method Neural Surrogate (`redoxquant.ml.surrogate`)
- Polynomial basis Ridge regressor predicting equivalent legacy assay values (ELISA $\text{OD}_{450}$, Octet BLI response, Protein A HPLC peak area) directly from Amperia signals with closed-form prediction uncertainty bands.

### 7. Qwen AI Bioprocess Copilot (`redoxquant.ai`)
- Full OpenAI / Qwen function-calling tool specifications.
- Conversational bioprocess intelligence connecting to local Qwen2.5 endpoints (via Ollama, vLLM, LM Studio) or running offline via deterministic rule-based `DiagnosticReasoner`.

---

## Installation

```bash
# Core package
pip install redoxquant

# With Machine Learning & Excel support
pip install "redoxquant[all]"

# Editable installation for development & testing
git clone https://github.com/lynchaos/redoxquant.git
cd redoxquant
pip install -e ".[dev]"
```

---

## Quickstart

```python
import redoxquant as rq

# 1. Ingest Amperia analysis export into a clean canonical DataFrame
df = rq.read_csv("analysis_export.csv")

# 2. Check for microfluidic bubbles or electrode anomalies
anomaly_report = rq.ml.detect_sensor_anomalies(df)
clean_df = anomaly_report.cleaned_df

# 3. Fit standard curve (5PL, variance-weighted 1/y^2)
curve = rq.fit_calibration(clean_df["concentration"], clean_df["signal_ru"], model="5PL")

# 4. Quantify samples with Delta-Method 95% Confidence Intervals
quant_df = rq.quantify(clean_df, curve)
sample_signal = 1187.4
ci = curve.back_calculate_with_ci(sample_signal, ci=0.95)
print(f"Concentration: {ci.estimate:.2f} µg/mL (95% CI: [{ci.lower:.2f}, {ci.upper:.2f}])")

# 5. Evaluate Bioanalytical Limits & Total Error Profile
limits = rq.compute_assay_limits(clean_df, curve)
print(f"Dynamic Range [LLOQ, ULOQ]: [{limits.lloq:.2f}, {limits.uloq:.2f}] µg/mL")

# 6. Compare against legacy ELISA results
deming = rq.deming_regression(elisa_concentrations, amperia_concentrations)
print(f"Deming Slope: {deming.slope:.3f} (95% CI: [{deming.slope_ci[0]:.3f}, {deming.slope_ci[1]:.3f}])")

# 7. Generate Standalone Publication-Ready HTML Report
rq.generate_html_report(clean_df, curve, output_path="assay_run_report.html")
```

---

## Few-Shot Bayesian Calibration

Save $>80\%$ of calibration reagents by leveraging historical assay priors:

```python
from redoxquant.ml import Bayesian5PLPrior, fit_bayesian_5pl

# Fit with only 3 standard concentrations (e.g. 1.0, 25.0, 100.0 µg/mL)
prior = Bayesian5PLPrior.default_descending()
bayes_curve = fit_bayesian_5pl(few_shot_conc, few_shot_signal, prior=prior)

# Back-calculate with posterior Credible Interval
bayes_ci = bayes_curve.back_calculate_with_credible_interval(signal=1200.0, credible_interval=0.95)
print(f"Bayesian Estimate: {bayes_ci.estimate:.2f} µg/mL [{bayes_ci.lower:.2f}, {bayes_ci.upper:.2f}]")
```

---

## Scientific Validation Benchmarks

`redoxquant` includes a dedicated, peer-review-grade validation benchmark ([`tests/scientific_validation.py`](tests/scientific_validation.py)):

| Benchmark | Standard / Literature Reference | Target Criteria | `redoxquant` Result |
| :--- | :--- | :---: | :---: |
| **Real-World Instrument Fidelity** | Reconstructed Amperia Tocilizumab mAb export | Error $< 0.25\%$ | **Max Error: $0.181\%$** |
| **LBA Total Error Profile** | Azadeh et al. (2018), *AAPS J* | $\text{TE} \le 30\%$ ($40\%$ LLOQ) | **Max TE: $38.46\%$ (LLOQ)** |
| **Orthogonal Method Deming Slope** | Linnet (1993), *Clin Chem* | Unbiased ($1.050 \in 95\%\text{ CI}$) | **$\beta_1 = 1.055$ $[1.023, 1.086]$** |
| **Passing–Bablok Linearity** | Passing & Bablok (1983), *J Clin Chem* | CUSUM $p > 0.05$ | **$p = 0.573$ (Linear)** |
| **Bayesian Few-Shot Coverage** | $N=100$ Monte Carlo simulation | Empirical Coverage $\ge 90\%$ | **$100.0\%$ Coverage** |
| **Sensor Anomaly Specificity** | $N=200$ trials ($6,400$ probe tests) | False Alarm Rate $< 1.0\%$ | **$100.00\%$ Specificity ($0\%$ FPR)** |

To run the complete scientific validation suite:
```bash
python tests/scientific_validation.py
```

---

## Testing

The test suite contains 44 unit and integration tests covering the entire mathematical and ML stack:

```bash
pytest
# Output: 44 passed in 7.05s
```

## Slack Integration & Automated Lab Alerts

You can stream automated assay completion alerts, sensor anomaly warnings, and run reports directly into your **RedoxQuant Slack Workspace**:

```python
import redoxquant as rq

df = rq.read_csv("analysis_export.csv")
curve = rq.fit_calibration(df["concentration"], df["signal_ru"])

# Post structured run summary card to Slack
rq.send_slack_alert(
    webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    title="Bioreactor B3 - Tocilizumab Run #42 Complete",
    message="Standard curve fitted successfully with 0 anomaly flags.",
    df=df,
    curve=curve,
)
```

### Direct In-Python Developer Feedback

Have an assay format question, feature request, or dataset you want to discuss? Send feedback straight from your Python script or Jupyter notebook to the developer's Slack channel:

```python
import redoxquant as rq

# Direct line to the developer
rq.submit_feedback(
    "Would love support for batch multi-plate kinetics exports!",
    user_email="scientist@biotech.com",
    user_name="Dr. Elena",
    category="Feature Request",
)
```

---

## Author & Community Feedback

`redoxquant` was designed and developed by **Kemal Yaylalı** (Solo Developer).

I welcome feedback, collaborative discussions on bioprocess machine learning, and contributions:

* **Support & Inquiries**: [`support@yaylali.uk`](mailto:support@yaylali.uk)
* **Personal Website**: [https://kemal.yaylali.uk](https://kemal.yaylali.uk)
* **GitHub Issues**: [@lynchaos/redoxquant](https://github.com/lynchaos/redoxquant/issues)
* **Slack Community**: Direct notifications and community discussions via the RedoxQuant Slack Workspace.



---

## References

1. **Azadeh, M., et al.** (2018). *Calibration Curves in Quantitative Ligand Binding Assays: Recommendations and Best Practices for Preparation, Design, and Editing of Calibration Curves.* **The AAPS Journal**, 20(1), 22. [DOI: 10.1208/s12248-017-0159-4](https://doi.org/10.1208/s12248-017-0159-4)
2. **Gottschalk, P. G., & Dunn, J. R.** (2005). *The five-parameter logistic: a characterization and comparison with the four-parameter logistic.* **Analytical Biochemistry**, 343(1), 54–65. [DOI: 10.1016/j.ab.2005.04.035](https://doi.org/10.1016/j.ab.2005.04.035)
3. **Linnet, K.** (1993). *Evaluation of regression procedures for methods comparison studies.* **Clinical Chemistry**, 39(3), 424–432. [PMID: 8448852](https://pubmed.ncbi.nlm.nih.gov/8448852/)
4. **Passing, H., & Bablok, W.** (1983). *A New Biometrical Procedure for Testing the Equality of Measurements from Two Different Analytical Methods.* **J. Clin. Chem. Clin. Biochem.**, 21(11), 709–720. [DOI: 10.1515/cclm.1983.21.11.709](https://doi.org/10.1515/cclm.1983.21.11.709)
5. **Bland, J. M., & Altman, D.** (1986). *Statistical methods for assessing agreement between two methods of clinical measurement.* **The Lancet**, 327(8476), 307–310. [DOI: 10.1016/S0140-6736(86)90837-8](https://doi.org/10.1016/S0140-6736(86)90837-8)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
