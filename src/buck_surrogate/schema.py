"""CSV loading and physical-domain validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import FEATURE_COLUMNS, OPTIONAL_COLUMNS, TARGET_COLUMNS


@dataclass
class ValidationReport:
    rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def validate_dataframe(
    frame: pd.DataFrame,
    *,
    require_targets: bool,
) -> ValidationReport:
    report = ValidationReport(rows=len(frame))
    required = FEATURE_COLUMNS + (TARGET_COLUMNS if require_targets else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        report.errors.append(f"Missing required columns: {', '.join(missing)}")
        return report

    if frame.empty:
        report.errors.append("The CSV contains no data rows.")
        return report

    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        report.errors.append(f"Duplicated columns: {', '.join(duplicates)}")

    numeric_columns = required
    converted = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    bad_numeric = converted.isna().sum()
    for column, count in bad_numeric.items():
        if count:
            report.errors.append(f"{column}: {int(count)} missing or non-numeric values")

    finite_mask = np.isfinite(converted.to_numpy(dtype=float))
    if not finite_mask.all():
        report.errors.append("The required numeric columns contain infinite values.")

    if report.errors:
        return report

    for column in required:
        count = int((converted[column] <= 0.0).sum())
        if count:
            report.errors.append(f"{column}: {count} values must be greater than zero")

    invalid_conversion = converted["vout_v"] >= converted["vin_v"]
    if invalid_conversion.any():
        report.errors.append(
            f"vout_v must be lower than vin_v in {int(invalid_conversion.sum())} rows"
        )

    if require_targets:
        high_ripple = converted["ripple_current_a"] > converted["iout_a"]
        if high_ripple.any():
            report.warnings.append(
                f"{int(high_ripple.sum())} rows have ripple_current_a > iout_a; "
                "review CCM assumptions or measurement extraction."
            )

    known = set(FEATURE_COLUMNS + TARGET_COLUMNS + OPTIONAL_COLUMNS)
    unknown = [column for column in frame.columns if column not in known]
    if unknown:
        report.warnings.append(
            "Extra columns will be ignored by the model: " + ", ".join(unknown)
        )

    return report


def validate_or_raise(frame: pd.DataFrame, *, require_targets: bool) -> ValidationReport:
    report = validate_dataframe(frame, require_targets=require_targets)
    if not report.valid:
        raise ValueError("; ".join(report.errors))
    return report
