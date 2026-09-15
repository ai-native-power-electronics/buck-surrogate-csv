"""Learning-curve evaluation with a fixed held-out test set."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import FEATURE_COLUMNS, TARGET_COLUMNS
from .model import _target_metrics, build_estimator
from .schema import validate_or_raise


def default_training_sizes(maximum_rows: int) -> list[int]:
    """Return useful nested sizes while always including the full training pool."""
    if maximum_rows < 100:
        raise ValueError("At least 100 training rows are required for a learning curve.")

    candidates = [500, 1000, 2500, 5000, 10000, 20000, 50000, 100000]
    sizes = [size for size in candidates if size <= maximum_rows]
    if not sizes:
        sizes = [100]
    if sizes[-1] != maximum_rows:
        sizes.append(maximum_rows)
    return sizes


def _normalise_sizes(sizes: list[int] | None, maximum_rows: int) -> list[int]:
    if sizes is None:
        return default_training_sizes(maximum_rows)

    normalised = sorted(set(sizes))
    if not normalised:
        raise ValueError("At least one training size is required.")
    if normalised[0] < 100:
        raise ValueError("Each learning-curve training size must be at least 100.")
    if normalised[-1] > maximum_rows:
        raise ValueError(
            f"Requested {normalised[-1]} training rows, but only {maximum_rows} "
            "remain after creating the fixed test set."
        )
    return normalised


def evaluate_learning_curve(
    frame: pd.DataFrame,
    *,
    sizes: list[int] | None = None,
    test_fraction: float = 0.20,
    repeats: int = 3,
    seed: int = 42,
    max_iter: int = 250,
    learning_rate: float = 0.08,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Evaluate nested training subsets against one unchanged test set."""
    validate_or_raise(frame, require_targets=True)
    if not 0.05 <= test_fraction <= 0.50:
        raise ValueError("test_fraction must be between 0.05 and 0.50")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if len(frame) < 200:
        raise ValueError("At least 200 valid rows are required for a learning curve.")

    x = frame[FEATURE_COLUMNS].astype(float)
    y = frame[TARGET_COLUMNS].astype(float)
    x_pool, x_test, y_pool, y_test = train_test_split(
        x,
        y,
        test_size=test_fraction,
        random_state=seed,
    )
    training_sizes = _normalise_sizes(sizes, len(x_pool))

    records: list[dict] = []
    for repeat in range(repeats):
        repeat_seed = seed + repeat
        rng = np.random.default_rng(repeat_seed)
        order = rng.permutation(len(x_pool))

        for training_rows in training_sizes:
            subset = order[:training_rows]
            estimator = build_estimator(
                seed=repeat_seed,
                max_iter=max_iter,
                learning_rate=learning_rate,
            )
            fit_start = perf_counter()
            estimator.fit(x_pool.iloc[subset], y_pool.iloc[subset])
            fit_seconds = perf_counter() - fit_start

            predict_start = perf_counter()
            prediction = estimator.predict(x_test)
            predict_seconds = perf_counter() - predict_start

            for target_index, target in enumerate(TARGET_COLUMNS):
                record = {
                    "training_rows": training_rows,
                    "repeat": repeat + 1,
                    "seed": repeat_seed,
                    "target": target,
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                }
                record.update(
                    _target_metrics(
                        y_test[target].to_numpy(),
                        prediction[:, target_index],
                    )
                )
                records.append(record)

    raw = pd.DataFrame.from_records(records)
    metric_columns = [
        "r2",
        "mae",
        "rmse",
        "mape_pct",
        "p95_ape_pct",
        "fit_seconds",
        "predict_seconds",
    ]
    summary = (
        raw.groupby(["training_rows", "target"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary = summary.fillna(0.0)

    metadata = {
        "rows_total": len(frame),
        "rows_training_pool": len(x_pool),
        "rows_fixed_test": len(x_test),
        "test_fraction": test_fraction,
        "training_sizes": training_sizes,
        "repeats": repeats,
        "base_seed": seed,
        "max_iter": max_iter,
        "learning_rate": learning_rate,
        "test_policy": "one fixed random-row holdout for every curve point",
        "subset_policy": "nested random training subsets within each repeat",
    }
    return raw, summary, metadata


def summary_as_json(summary: pd.DataFrame, metadata: dict) -> dict:
    """Convert NumPy-backed values to JSON-safe built-in values."""
    return {
        "metadata": metadata,
        "points": json.loads(summary.to_json(orient="records")),
    }


def save_learning_curve_plot(summary: pd.DataFrame, path: str | Path) -> None:
    """Plot R-squared and MAPE against the number of training rows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colours = {"loss_w": "#176b87", "ripple_current_a": "#d1495b"}
    labels = {"loss_w": "Total loss", "ripple_current_a": "Ripple current"}

    for target in TARGET_COLUMNS:
        target_rows = summary[summary["target"] == target].sort_values("training_rows")
        x = target_rows["training_rows"].to_numpy()
        colour = colours[target]

        axes[0].plot(
            x,
            target_rows["r2_mean"],
            marker="o",
            label=labels[target],
            color=colour,
        )
        axes[0].fill_between(
            x,
            target_rows["r2_mean"] - target_rows["r2_std"],
            target_rows["r2_mean"] + target_rows["r2_std"],
            alpha=0.15,
            color=colour,
        )
        axes[1].plot(
            x,
            target_rows["mape_pct_mean"],
            marker="o",
            label=labels[target],
            color=colour,
        )
        axes[1].fill_between(
            x,
            np.maximum(0.0, target_rows["mape_pct_mean"] - target_rows["mape_pct_std"]),
            target_rows["mape_pct_mean"] + target_rows["mape_pct_std"],
            alpha=0.15,
            color=colour,
        )

    axes[0].set_title("Held-out R-squared")
    axes[0].set_ylabel("R-squared")
    axes[1].set_title("Held-out mean absolute percentage error")
    axes[1].set_ylabel("MAPE (%)")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Training rows")
        axis.grid(alpha=0.2)
        axis.legend()

    figure.suptitle("Learning curve — fixed test set")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
