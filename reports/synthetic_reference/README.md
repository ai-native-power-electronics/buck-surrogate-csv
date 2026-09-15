# Synthetic reference run

This folder records a reproduced end-to-end check of version 0.2.0 on 2026-09-06.

## Conditions

- Data source: bundled simplified equations
- Samples: 20,000
- Training rows: 16,000
- Held-out test rows: 4,000
- Split: random row holdout
- Seed: 42
- Python: 3.12.14
- NumPy: 2.5.3
- pandas: 2.3.3
- scikit-learn: 1.9.0

## Results

| Target | R-squared | MAE | MAPE | 95th-percentile APE |
|---|---:|---:|---:|---:|
| Total loss | 0.9996 | 0.0348 W | 1.52% | 4.77% |
| Ripple current | 0.9943 | 0.0644 A | 5.58% | 15.55% |

Batch inference over 10,000 rows had a median measured duration of 28.7 ms across 15 repetitions, or approximately 2.87 microseconds per row. This measurement does not include CSV I/O and is not a comparison against any simulator.

See `metrics.json` and `benchmark.json` for unrounded values. `parity_plots.png` shows every held-out prediction.

## Learning-curve result

The learning curve uses the same fixed 4,000-row test set at every point and nested training subsets. Each point is repeated three times. The final two points show that additional data still help, but with diminishing returns:

| Target | MAPE at 10,000 training rows | MAPE at 16,000 training rows |
|---|---:|---:|
| Total loss | 1.60% | 1.53% |
| Ripple current | 5.96% | 5.64% |

`learning_curve.csv` contains the aggregated values, `learning_curve_raw.csv` preserves every repetition, and `learning_curve.png` shows the trend. This curve measures sample-size behaviour only for the bundled synthetic distribution; it does not establish how many simulator or laboratory cases another problem requires.

> These results only describe approximation of the bundled equations inside the sampled domain. They are not experimental validation and should not be transferred to a different data source.
