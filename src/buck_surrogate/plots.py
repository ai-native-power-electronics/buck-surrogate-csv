"""Small reporting plots kept separate from model logic."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .constants import TARGET_COLUMNS


def save_parity_plots(predictions: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, len(TARGET_COLUMNS), figsize=(11, 4.5))
    for axis, target in zip(axes, TARGET_COLUMNS):
        reference = predictions[f"reference_{target}"]
        predicted = predictions[f"predicted_{target}"]
        lower = min(reference.min(), predicted.min())
        upper = max(reference.max(), predicted.max())
        axis.scatter(reference, predicted, s=10, alpha=0.35, edgecolors="none")
        axis.plot([lower, upper], [lower, upper], color="#d1495b", linewidth=1.5)
        axis.set_title(target)
        axis.set_xlabel("Reference")
        axis.set_ylabel("Prediction")
        axis.grid(alpha=0.2)

    figure.suptitle("Held-out test set: prediction vs reference")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
