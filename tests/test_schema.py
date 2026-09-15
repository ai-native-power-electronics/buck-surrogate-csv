from buck_surrogate.physics import synthetic_dataset
from buck_surrogate.schema import validate_dataframe


def test_valid_synthetic_data_passes_schema():
    report = validate_dataframe(synthetic_dataset(100, seed=1), require_targets=True)
    assert report.valid


def test_missing_target_fails_training_schema():
    frame = synthetic_dataset(100, seed=1).drop(columns=["loss_w"])
    report = validate_dataframe(frame, require_targets=True)
    assert not report.valid
    assert "loss_w" in report.errors[0]


def test_non_buck_voltage_is_rejected():
    frame = synthetic_dataset(100, seed=1)
    frame.loc[0, "vout_v"] = frame.loc[0, "vin_v"]
    report = validate_dataframe(frame, require_targets=True)
    assert not report.valid


def test_extra_metadata_is_only_a_warning():
    frame = synthetic_dataset(100, seed=1)
    frame["ambient_c"] = 25.0
    report = validate_dataframe(frame, require_targets=True)
    assert report.valid
    assert report.warnings
