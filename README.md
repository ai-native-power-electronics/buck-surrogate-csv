# Buck Converter Surrogate — CSV-first workflow

A reproducible, source-agnostic workflow for building a machine-learning surrogate of a Buck converter. The included demo uses transparent, simplified equations. You can replace that dataset with results from SPICE, PLECS, PSIM, MATLAB/Simulink, another simulator, or laboratory measurements without changing the training pipeline.

> [!IMPORTANT]
> The bundled synthetic data are a teaching reference, not experimental ground truth. Training a model on closed-form equations demonstrates the surrogate workflow; it does not prove an accuracy or speed advantage over those equations. Reported metrics and inference times are generated locally and are never hard-coded.

## What this project demonstrates

- A documented CSV contract with units in every column name.
- Reproducible synthetic sampling with a fixed random seed.
- Validation before training, including physical-domain checks.
- A held-out test split and per-output error metrics.
- Saved training bounds and out-of-domain warnings at prediction time.
- Measured surrogate inference time without inventing a simulator baseline.
- Design-space exploration subject to a ripple-current constraint.

## Quick start on Windows

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Generate the transparent synthetic demonstration dataset:

```powershell
buck-surrogate generate --output data/synthetic_buck.csv --samples 20000 --seed 42
```

Validate and train:

```powershell
buck-surrogate validate --input data/synthetic_buck.csv
buck-surrogate train --input data/synthetic_buck.csv --output-dir outputs
```

Measure whether additional training rows are still improving the model:

```powershell
buck-surrogate learning-curve --input data/synthetic_buck.csv --output-dir outputs/learning_curve
```

The command keeps one test set unchanged, trains on nested subsets of increasing size and repeats the experiment three times by default. You can supply explicit training sizes with `--sizes 500 1000 2500 5000 10000`.

Measure inference performance on your computer:

```powershell
buck-surrogate benchmark --model outputs/buck_surrogate.joblib --rows 10000
```

Explore candidate designs:

```powershell
buck-surrogate explore --model outputs/buck_surrogate.joblib --candidates 100000 --max-ripple-ratio 0.30 --output outputs/top_candidates.csv
```

Predict your own design points:

```powershell
buck-surrogate predict --model outputs/buck_surrogate.joblib --input data/designs_to_predict_template.csv --output outputs/predictions.csv
```

## Bring your own CSV

For training, preserve these required columns:

| Role | Column | Unit | Meaning |
|---|---|---:|---|
| Input | `vin_v` | V | Input voltage |
| Input | `vout_v` | V | Output voltage |
| Input | `iout_a` | A | Output current |
| Input | `fsw_hz` | Hz | Switching frequency |
| Input | `inductance_h` | H | Buck inductance |
| Target | `loss_w` | W | Total converter loss from your chosen source |
| Target | `ripple_current_a` | A pk-pk | Inductor current ripple from your chosen source |

`case_id` and `source` are optional metadata columns. Unknown columns are preserved in the source file but ignored by the model. See [`docs/METHOD.md`](docs/METHOD.md) for extraction and validation guidance.

Example sources:

- `source=synthetic_equations`
- `source=ltspice`
- `source=plecs`
- `source=psim`
- `source=simulink`
- `source=lab`

The training and prediction commands do not call a simulator. Your simulator only needs to export one row per operating point using the schema above.

## Outputs

`buck-surrogate train` creates:

- `outputs/buck_surrogate.joblib`: trained estimator plus schema and training bounds.
- `outputs/metrics.json`: test metrics, data split and model settings.
- `outputs/predictions_test.csv`: held-out references and predictions.
- `outputs/parity_plots.png`: prediction-versus-reference plots.
- `outputs/learning_curve/learning_curve.csv`: mean and standard deviation by target and training size.
- `outputs/learning_curve/learning_curve_raw.csv`: every individual repetition.
- `outputs/learning_curve/learning_curve.json`: machine-readable configuration and results.
- `outputs/learning_curve/learning_curve.png`: R-squared and MAPE trends.

The model predicts `loss_w` and `ripple_current_a`. Efficiency is calculated afterward from predicted loss:

```text
efficiency = Pout / (Pout + loss)
```

This keeps the two outputs physically consistent.

## Reproduced reference run

The workflow was executed end to end on 20,000 equation-generated rows with seed 42. The saved metrics, benchmark conditions and parity plot are available in [`reports/synthetic_reference`](reports/synthetic_reference). They are an example run, not a performance guarantee for simulator or laboratory data.

## Honest reporting checklist

Before publishing a result:

1. Name the data source and number of valid operating points.
2. State the sampled design limits and test-split method.
3. Report each target separately; do not hide a weak target behind an average score.
4. Use a learning curve rather than assuming a row count is sufficient.
5. Call synthetic values “reference data”, not “ground truth”.
6. Measure inference and simulation times on the same machine if claiming a speed-up.
7. Test the selected design again in the original simulator or laboratory.
8. Keep predictions inside the training domain unless extrapolation is explicitly studied.

## Project structure

```text
data/                 CSV templates and generated datasets
docs/METHOD.md        step-by-step method and simulator-data guidance
src/buck_surrogate/   generator, validation, model and CLI
tests/                physics, schema and end-to-end tests
outputs/              generated models, metrics and figures
reports/              traceable example results from a reproduced run
```

## Current scope

The synthetic generator represents a simplified CCM synchronous Buck converter with fixed, generic loss parameters. It is intentionally not tied to any part number. It omits thermal dynamics, magnetic saturation, layout parasitics, control-loop dynamics and detailed switching waveforms. Those effects can be represented by replacing the synthetic CSV with higher-fidelity data.

## License

Released under the [MIT License](LICENSE). The synthetic equations and trained surrogate are engineering demonstrations, not hardware design approval.
