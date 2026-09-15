"""Shared dataset schema."""

FEATURE_COLUMNS = [
    "vin_v",
    "vout_v",
    "iout_a",
    "fsw_hz",
    "inductance_h",
]

TARGET_COLUMNS = [
    "loss_w",
    "ripple_current_a",
]

OPTIONAL_COLUMNS = [
    "case_id",
    "source",
]
