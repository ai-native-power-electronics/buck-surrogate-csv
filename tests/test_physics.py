import numpy as np

from buck_surrogate.physics import synthetic_dataset


def test_synthetic_dataset_is_reproducible_and_physical():
    first = synthetic_dataset(250, seed=7)
    second = synthetic_dataset(250, seed=7)

    assert first.equals(second)
    assert (first["vout_v"] < first["vin_v"]).all()
    assert (first["loss_w"] > 0.0).all()
    assert (first["ripple_current_a"] > 0.0).all()
    assert (first["ripple_current_a"] <= first["iout_a"]).all()
    assert np.isfinite(first.select_dtypes(include="number").to_numpy()).all()
