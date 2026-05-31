"""Rectified-flow native sigma schedule builders."""

from __future__ import annotations

from math import exp, isfinite

from .schedule_math import validate_steps


def build_flow_rf_linear_shift_sigmas(
    steps: int,
    *,
    shift: float = 5.0,
    num_train_timesteps: int = 1000,
) -> list[float]:
    """Build Cosmos 2.5's normalized RF linear sigma schedule with shift."""

    validate_steps(steps)
    shift = float(shift)
    if not isfinite(shift) or shift <= 0.0:
        raise ValueError("shift must be finite and positive")
    if num_train_timesteps < 2:
        raise ValueError("num_train_timesteps must be at least 2")

    sigma_max = 1.0 - (1.0 / float(num_train_timesteps))
    values = []
    for step in range(steps):
        sigma = sigma_max * (1.0 - step / steps)
        values.append(_rf_shift_sigma(sigma, shift=shift))
    values.append(0.0)
    return values


def build_flow_diffusers_linear_shift_sigmas(
    steps: int,
    *,
    shift: float = 3.0,
    num_train_timesteps: int = 1000,
) -> list[float]:
    """Build Diffusers FlowMatchEulerDiscreteScheduler sigmas.

    This mirrors the non-dynamic-shifting ``set_timesteps`` path used by
    Diffusers for Anima-Base-v1.0-Diffusers.
    """

    validate_steps(steps)
    shift = float(shift)
    if not isfinite(shift) or shift <= 0.0:
        raise ValueError("shift must be finite and positive")
    if num_train_timesteps < 2:
        raise ValueError("num_train_timesteps must be at least 2")

    training_sigma_min = _rf_shift_sigma(
        1.0 / float(num_train_timesteps),
        shift=shift,
    )
    values = []
    for step in range(steps):
        alpha = step / (steps - 1) if steps > 1 else 0.0
        sigma = 1.0 + alpha * (training_sigma_min - 1.0)
        values.append(_rf_shift_sigma(sigma, shift=shift))
    values.append(0.0)
    return values


def build_flow_rf_linear_s_tail_shift5_sigmas(
    steps: int,
    *,
    center: float = 0.94,
    width: float = 0.04,
    num_train_timesteps: int = 1000,
) -> list[float]:
    """Build a fixed shift-5 RF linear schedule with an S-shaped tail."""

    validate_steps(steps)
    center = float(center)
    width = float(width)
    if not isfinite(center) or not (0.0 < center < 1.0):
        raise ValueError("center must be finite and in the range (0, 1)")
    if not isfinite(width) or width <= 0.0:
        raise ValueError("width must be finite and positive")

    base = build_flow_rf_linear_shift_sigmas(
        steps,
        shift=5.0,
        num_train_timesteps=num_train_timesteps,
    )
    gate_start = _stable_sigmoid(center / width)
    gate_end = _stable_sigmoid((center - 1.0) / width)
    gate_denom = gate_start - gate_end
    if gate_denom <= 0.0:
        raise ValueError("invalid sigmoid gate parameters")

    values = []
    for step in range(steps):
        progress = step / steps
        raw_gate = _stable_sigmoid((center - progress) / width)
        gate = (raw_gate - gate_end) / gate_denom
        values.append(base[step] * min(1.0, max(0.0, gate)))
    values.append(0.0)
    return values


def _rf_shift_sigma(sigma: float, *, shift: float) -> float:
    return float(shift) * float(sigma) / (1.0 + (float(shift) - 1.0) * float(sigma))


def _stable_sigmoid(value: float) -> float:
    value = float(value)
    if value >= 60.0:
        return 1.0
    if value <= -60.0:
        return 0.0
    return 1.0 / (1.0 + exp(-value))
