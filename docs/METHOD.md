# Step-by-step method

## 1. Define the question before generating data

This project asks a narrow question:

> Given a Buck operating point and inductance, can a supervised model estimate total loss and inductor-current ripple inside a documented design domain?

The objective is a reusable workflow, not a claim that machine learning should replace equations when equations are already fast and sufficient.

## 2. Use one stable CSV contract

Training files require:

```text
vin_v,vout_v,iout_a,fsw_hz,inductance_h,loss_w,ripple_current_a
```

Optional traceability fields:

```text
case_id,source
```

Keep SI units in the stored CSV. Convert microhenries, kilohertz and milliohms before export. Each row must represent one valid steady-state operating point.

## 3. Choose the data source

### Synthetic demonstration

The built-in generator uses simplified CCM equations and generic, fixed loss parameters. Its purpose is to make the complete workflow runnable by anyone. It rejects points where peak-to-peak ripple exceeds output current, keeping the demonstration conservatively inside CCM. The assumptions are visible in `src/buck_surrogate/physics.py`.

### Simulator data

Sweep the five input variables, measure total loss and peak-to-peak inductor-current ripple after steady state, and export one row per valid simulation. Failed simulations should not silently become zeros; record them separately and diagnose their region of the design space.

Recommended flow:

1. Create the design matrix with Latin hypercube, Sobol or another space-filling sampler.
2. Run the simulator with a consistent settling time and measurement window.
3. Check energy signs and units.
4. Export successful cases to the required CSV columns.
5. Retain the simulator input deck and extraction script for reproducibility.

### Laboratory data

Use the same columns, but also retain instrument configuration, calibration date, ambient conditions and uncertainty outside the model CSV. Repeated measurements should have distinct `case_id` values rather than being averaged without traceability.

## 4. Validate before training

Run:

```powershell
buck-surrogate validate --input path\to\your_data.csv
```

Validation rejects missing, non-numeric, non-finite or non-positive values and points where `vout_v >= vin_v`. High ripple relative to output current is reported as a warning because the simplified CCM interpretation may be weak there.

## 5. Split and train

The default model is a separate histogram gradient-boosting regressor for each target. A fixed random split makes the example reproducible.

For a stronger technical study, do not rely only on a random row split. Add one or more of these tests:

- Hold out complete voltage or load regions.
- Hold out component/design combinations.
- Train on one simulator model and test on another.
- Train on simulations and test on measured hardware.

## 6. Evaluate each output separately

The generated report includes:

- R-squared (`r2`)
- Mean absolute error (`mae`)
- Root mean squared error (`rmse`)
- Mean absolute percentage error (`mape_pct`)
- 95th-percentile absolute percentage error (`p95_ape_pct`)

Inspect parity plots and worst cases. A high aggregate score is not enough if the error is concentrated near important design constraints.

## 7. Check whether the dataset is large enough

Run:

```powershell
buck-surrogate learning-curve --input path\to\your_data.csv
```

The command creates one fixed held-out test set, then trains on nested subsets of the remaining rows. Three repetitions are used by default to expose sensitivity to subset selection. If held-out error is still falling materially at the largest size, more representative data may help. If it has plateaued at an unacceptable error, adding rows from the same distribution may not solve the problem; inspect missing inputs, regime changes, label quality and model bias.

A plateau does not prove broad validity. Repeat the analysis with grouped or boundary holdouts when rows share a physical design, component family or simulation campaign.

## 8. Benchmark honestly

`buck-surrogate benchmark` warms up the model and measures repeated batch inference. It reports median batch time, time per row and rows per second.

It does not manufacture a speed-up number. To claim a comparison against a simulator, independently time the simulator on the same computer, disclose whether startup and file I/O are included, and compare equivalent workloads.

## 9. Explore, then verify

The exploration command samples candidates inside the training bounds, predicts both targets, filters by a maximum ripple ratio and ranks feasible points by predicted loss. This is screening, not final validation.

Always rerun the shortlisted designs in the original data source—equations, detailed simulator or laboratory—before drawing an engineering conclusion.

## 10. Suggested extensions

- Add temperature, device parameters and magnetic geometry as inputs.
- Use grouped or boundary holdouts.
- Add prediction intervals or conformal calibration.
- Compare direct ML against a physics-plus-residual model.
- Add active learning to select the next expensive simulation points.
- Quantify domain shift between idealized, vendor-model and measured data.
