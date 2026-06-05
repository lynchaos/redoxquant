# redoxquant

[![CI](https://github.com/lynchaos/redoxquant/actions/workflows/ci.yml/badge.svg)](https://github.com/lynchaos/redoxquant/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An open Python companion for **Amperia™** protein-quantification analysis exports.

`redoxquant` works strictly *downstream* of the instrument's exported file. It
parses Amperia analysis exports into tidy data, refits the 5-parameter logistic
(5PL) standard curve the instrument uses — ascending or descending — and adds
the rigour a benchtop GUI typically leaves out: back-calculation with the same
curve, replicate CV, dilution handling, and QC flags. The goal is to extend
where Amperia data can *go* — notebooks, pipelines, ELNs, batch jobs — not to
replace the instrument software.

> **Not affiliated with or endorsed by Abselion / HexagonFab Ltd.** "Amperia"
> and "Abselion" are trademarks of their owner, used here only to describe file
> compatibility. The library never touches raw electrode signals or any
> proprietary signal-to-RU conversion; it consumes Response Units (RU) as
> exported. For Research Use Only.

## Why

Amperia exports are CSV / XLSX / PDF, transferred by USB. The CSV/XLSX carry,
per probe reading: `Signal` (RU), `Signal Compensation`, `Concentration`,
`Adjusted Concentration`, `Dilution Factor`, and replicate `Tags`. That's a
clean, scriptable substrate — but you have to leave the touchscreen to do
anything programmatic with it. This library is that programmatic layer.

## Install

```bash
pip install -e ".[dev]"      # from a clone, with test extras
```

## Quickstart

```python
import redoxquant as rq

df = rq.read_csv("analysis_export.csv")          # tidy canonical frame
curve = rq.fit_calibration(conc, signal)         # 5PL, auto-detects orientation
q = rq.quantify(df, curve)                        # adds *_calc concentration columns
stats = rq.group_stats(df)                        # per-tag mean / std / %CV
flagged = rq.qc_flags(df)                          # compensation & CV flags
```

See [`examples/quickstart.py`](examples/quickstart.py) for an end-to-end run on
the bundled demo data.

## Fidelity to the instrument

The test suite includes a real worked example reconstructed from an Amperia
analysis export (a Tocilizumab quantification, tags `mid`/`hi`). Refitting the
descending 5PL from the export's own (concentration, signal) points and
back-calculating reproduces the instrument's **reported concentrations to
within ~0.1%**, and its **adjusted concentrations to within ~0.2%**, with the
descending orientation detected automatically.

This demonstrates the library's inverse agrees with the instrument's forward
model on real numbers. Note the demo screenshot did not include the explicit
*standard* rows, so this is round-trip consistency on sample-derived points
rather than a reconstruction of the exact fitted curve; exact curve
reproduction awaits a full export containing the standard rows. The
signal-compensation semantics (a multiplicative correction applied to signal
before the curve) are likewise inferred from the export layout and flagged in
the code as an assumption to confirm.

## Roadmap

- **Now (v0.0.x):** parser, 4PL/5PL fit (both orientations), quantification,
  replicate stats, basic QC, synthetic data generator.
- **Next:** ~~confidence intervals on back-calculated concentrations~~ ✓ (v0.0.2);
  LOD / LLOQ / ULOQ from standards; parallelism / dilutional-linearity using the
  `Dilution Factor` column; %RE and total-error per Azadeh et al. (2018).
- **Later:** ELISA method comparison (Deming, Passing–Bablok, Bland–Altman);
  one-call HTML/PDF run reports; signal-vs-step stability per sample.

## Calibration model

The 5PL follows the ligand-binding-assay best practices of
[Azadeh et al. (2018)](https://doi.org/10.1208/s12248-017-0159-4), the same
reference cited by the Amperia software:

```
y = d + (a - d) / (1 + (x / c) ** b) ** g
```

with `a` = response at x→0, `d` = response at x→∞, `c` = inflection, `b` = Hill
slope, `g` = asymmetry. A 4PL is the special case `g = 1`.

## License

MIT — see [LICENSE](LICENSE).
