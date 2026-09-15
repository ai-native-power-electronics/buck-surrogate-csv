"""Training, evaluation, persistence and prediction."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from . import __version__
from .constants import FEATURE_COLUMNS, TARGET_COLUMNS
from .schema import validate_or_raise


def _target_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    absolute_percentage_error = np.abs((prediction - reference) / reference) * 100.0
    return {
        "r2": float(r2_score(reference, prediction)),
        "mae": float(mean_absolute_error(reference, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(reference, prediction))),
        "mape_pct": float(np.mean(absolute_percentage_error)),
        "p95_ape_pct": float(np.percentile(absolute_percentage_error, 95)),
    }


def build_estimator(
    *,
    seed: int,
    max_iter: int,
    learning_rate: float,
) -> MultiOutputRegressor:
    """Build the estimator used by both final training and learning curves."""
    return MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            l2_regularization=1e-5,
            random_state=seed,
        )
    )


def train_model(
    frame: pd.DataFrame,
    *,
    test_fraction: float = 0.20,
    seed: int = 42,
    max_iter: int = 250,
    learning_rate: float = 0.08,
) -> tuple[dict, dict, pd.DataFrame]:
    """Train on a reproducible random split and return bundle, metrics and test rows."""
    validate_or_raise(frame, require_targets=True)
    if not 0.05 <= test_fraction <= 0.50:
        raise ValueError("test_fraction must be between 0.05 and 0.50")
    if len(frame) < 100:
        raise ValueError("At least 100 valid rows are required for training.")

    x = frame[FEATURE_COLUMNS].astype(float)
    y = frame[TARGET_COLUMNS].astype(float)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_fraction,
        random_state=seed,
    )

    estimator = build_estimator(
        seed=seed,
        max_iter=max_iter,
        learning_rate=learning_rate,
    )
    estimator.fit(x_train, y_train)
    prediction = estimator.predict(x_test)

    per_target = {
        target: _target_metrics(y_test[target].to_numpy(), prediction[:, index])
        for index, target in enumerate(TARGET_COLUMNS)
    }
    metrics = {
        "data": {
            "rows_total": len(frame),
            "rows_train": len(x_train),
            "rows_test": len(x_test),
            "test_fraction": test_fraction,
            "split": "random_row_holdout",
            "seed": seed,
        },
        "model": {
            "estimator": "MultiOutputRegressor(HistGradientBoostingRegressor)",
            "max_iter": max_iter,
            "learning_rate": learning_rate,
        },
        "targets": per_target,
    }

    input_bounds = {
        column: {"min": float(x_train[column].min()), "max": float(x_train[column].max())}
        for column in FEATURE_COLUMNS
    }
    bundle = {
        "package_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "features": FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "training_bounds": input_bounds,
        "estimator": estimator,
        "metrics": metrics,
    }

    test_predictions = x_test.copy()
    for column in TARGET_COLUMNS:
        test_predictions[f"reference_{column}"] = y_test[column]
    for index, column in enumerate(TARGET_COLUMNS):
        test_predictions[f"predicted_{column}"] = prediction[:, index]
    test_predictions = test_predictions.sort_index()

    return bundle, metrics, test_predictions


def save_bundle(bundle: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_bundle(path: str | Path) -> dict:
    bundle = joblib.load(Path(path))
    required_keys = {"features", "targets", "training_bounds", "estimator"}
    missing = required_keys.difference(bundle)
    if missing:
        raise ValueError(f"Invalid model bundle; missing: {', '.join(sorted(missing))}")
    return bundle


def predict_frame(bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    validate_or_raise(frame, require_targets=False)
    x = frame[bundle["features"]].astype(float)
    prediction = bundle["estimator"].predict(x)

    result = frame.copy()
    for index, target in enumerate(bundle["targets"]):
        result[f"predicted_{target}"] = prediction[:, index]

    predicted_loss = np.maximum(result["predicted_loss_w"].to_numpy(), 0.0)
    output_power = result["vout_v"].to_numpy() * result["iout_a"].to_numpy()
    result["predicted_efficiency_pct"] = 100.0 * output_power / (
        output_power + predicted_loss
    )

    inside = np.ones(len(result), dtype=bool)
    for column, bounds in bundle["training_bounds"].items():
        inside &= result[column].between(bounds["min"], bounds["max"]).to_numpy()
    result["within_training_domain"] = inside
    return result


def benchmark_model(
    bundle: dict,
    frame: pd.DataFrame,
    *,
    rows: int,
    repeats: int = 15,
    seed: int = 42,
) -> dict[str, float | int]:
    if rows < 1 or repeats < 3:
        raise ValueError("rows must be positive and repeats must be at least 3")

    rng = np.random.default_rng(seed)
    source = frame[bundle["features"]].astype(float)
    indices = rng.integers(0, len(source), size=rows)
    batch = source.iloc[indices]
    bundle["estimator"].predict(batch.iloc[: min(100, rows)])

    durations = []
    for _ in range(repeats):
        start = perf_counter()
        bundle["estimator"].predict(batch)
        durations.append(perf_counter() - start)

    median_seconds = float(np.median(durations))
    return {
        "rows": rows,
        "repeats": repeats,
        "median_batch_seconds": median_seconds,
        "median_microseconds_per_row": median_seconds * 1e6 / rows,
        "median_rows_per_second": rows / median_seconds,
    }
