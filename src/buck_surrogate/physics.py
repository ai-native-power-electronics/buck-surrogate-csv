"""Transparent synthetic reference generator for a simplified Buck converter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticBuckConfig:
    """Fixed generic parameters used only by the synthetic demonstration."""

    rds_on_high_ohm: float = 0.015
    rds_on_low_ohm: float = 0.008
    rise_time_s: float = 12e-9
    fall_time_s: float = 10e-9
    gate_charge_high_c: float = 35e-9
    gate_charge_low_c: float = 25e-9
    gate_drive_v: float = 10.0
    dead_time_s: float = 50e-9
    body_diode_v: float = 0.75
    inductor_dcr_ohm: float = 0.020


def sample_inputs(samples: int, seed: int) -> pd.DataFrame:
    """Sample generic Buck operating points in a conservative CCM region."""
    if samples < 1:
        raise ValueError("samples must be at least 1")

    rng = np.random.default_rng(seed)
    accepted: list[pd.DataFrame] = []
    accepted_rows = 0
    while accepted_rows < samples:
        batch_size = max(256, 2 * (samples - accepted_rows))
        vin = rng.uniform(12.0, 60.0, batch_size)
        vout_upper = np.minimum(24.0, 0.85 * vin)
        vout = rng.uniform(3.3, vout_upper)
        iout = rng.uniform(0.5, 15.0, batch_size)
        fsw = np.exp(rng.uniform(np.log(50e3), np.log(500e3), batch_size))
        inductance = np.exp(
            rng.uniform(np.log(10e-6), np.log(150e-6), batch_size)
        )
        duty = vout / vin
        ripple = (vin - vout) * duty / (inductance * fsw)

        # Keep the synthetic demonstration comfortably inside CCM. The real
        # CSV validator still accepts wider domains from higher-fidelity sources.
        keep = ripple <= iout
        batch = pd.DataFrame(
            {
                "vin_v": vin[keep],
                "vout_v": vout[keep],
                "iout_a": iout[keep],
                "fsw_hz": fsw[keep],
                "inductance_h": inductance[keep],
            }
        )
        accepted.append(batch)
        accepted_rows += len(batch)

    return pd.concat(accepted, ignore_index=True).head(samples)


def evaluate_reference(
    inputs: pd.DataFrame,
    config: SyntheticBuckConfig | None = None,
) -> pd.DataFrame:
    """Evaluate simplified CCM ripple and generic loss equations."""
    cfg = config or SyntheticBuckConfig()

    vin = inputs["vin_v"].to_numpy(dtype=float)
    vout = inputs["vout_v"].to_numpy(dtype=float)
    iout = inputs["iout_a"].to_numpy(dtype=float)
    fsw = inputs["fsw_hz"].to_numpy(dtype=float)
    inductance = inputs["inductance_h"].to_numpy(dtype=float)

    duty = vout / vin
    ripple = (vin - vout) * duty / (inductance * fsw)
    current_squared_mean = iout**2 + ripple**2 / 12.0

    conduction_high = duty * current_squared_mean * cfg.rds_on_high_ohm
    conduction_low = (1.0 - duty) * current_squared_mean * cfg.rds_on_low_ohm
    switching_high = (
        0.5 * vin * iout * (cfg.rise_time_s + cfg.fall_time_s) * fsw
    )
    gate_drive = (
        (cfg.gate_charge_high_c + cfg.gate_charge_low_c) * cfg.gate_drive_v * fsw
    )
    dead_time = 2.0 * cfg.dead_time_s * fsw * cfg.body_diode_v * iout
    inductor_copper = current_squared_mean * cfg.inductor_dcr_ohm

    total_loss = (
        conduction_high
        + conduction_low
        + switching_high
        + gate_drive
        + dead_time
        + inductor_copper
    )

    return pd.DataFrame(
        {
            "loss_w": total_loss,
            "ripple_current_a": ripple,
        },
        index=inputs.index,
    )


def synthetic_dataset(samples: int, seed: int = 42) -> pd.DataFrame:
    """Create a traceable synthetic dataset using the documented equations."""
    inputs = sample_inputs(samples=samples, seed=seed)
    targets = evaluate_reference(inputs)
    result = pd.concat([inputs, targets], axis=1)
    result.insert(0, "source", "synthetic_equations")
    width = max(6, len(str(samples)))
    result.insert(0, "case_id", [f"syn_{i:0{width}d}" for i in range(samples)])
    return result
