"""Candidate screening inside the model training domain."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import predict_frame


def _log_uniform(rng: np.random.Generator, low: float, high: float, size: int) -> np.ndarray:
    return np.exp(rng.uniform(np.log(low), np.log(high), size))


def explore_candidates(
    bundle: dict,
    *,
    candidates: int,
    max_ripple_ratio: float,
    seed: int,
    top: int,
) -> tuple[pd.DataFrame, int]:
    if candidates < 1 or top < 1:
        raise ValueError("candidates and top must be positive")
    if max_ripple_ratio <= 0.0:
        raise ValueError("max_ripple_ratio must be positive")

    bounds = bundle["training_bounds"]
    rng = np.random.default_rng(seed)
    vin = rng.uniform(bounds["vin_v"]["min"], bounds["vin_v"]["max"], candidates)
    vout_low = bounds["vout_v"]["min"]
    vout_high = np.minimum(bounds["vout_v"]["max"], 0.99 * vin)
    valid_voltage = vout_high > vout_low

    frame = pd.DataFrame(
        {
            "vin_v": vin[valid_voltage],
            "vout_v": rng.uniform(vout_low, vout_high[valid_voltage]),
            "iout_a": rng.uniform(
                bounds["iout_a"]["min"], bounds["iout_a"]["max"], valid_voltage.sum()
            ),
            "fsw_hz": _log_uniform(
                rng,
                bounds["fsw_hz"]["min"],
                bounds["fsw_hz"]["max"],
                valid_voltage.sum(),
            ),
            "inductance_h": _log_uniform(
                rng,
                bounds["inductance_h"]["min"],
                bounds["inductance_h"]["max"],
                valid_voltage.sum(),
            ),
        }
    )
    prediction = predict_frame(bundle, frame)
    prediction["predicted_ripple_ratio"] = (
        prediction["predicted_ripple_current_a"] / prediction["iout_a"]
    )
    feasible = prediction[
        (prediction["predicted_loss_w"] > 0.0)
        & (prediction["predicted_ripple_current_a"] > 0.0)
        & (prediction["predicted_ripple_ratio"] <= max_ripple_ratio)
        & prediction["within_training_domain"]
    ]
    ranked = feasible.sort_values("predicted_loss_w").head(top).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked, len(feasible)
