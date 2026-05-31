"""Table-derived and phase-based sigma schedule builders."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Sequence

from .schedule_math import as_float_list, linspace, validate_steps


@dataclass(frozen=True)
class PhaseSteps:
    """Number of transitions assigned to each denoising phase."""

    early: int
    mid: int
    late: int

    @property
    def total(self) -> int:
        return self.early + self.mid + self.late


def build_simple_sigmas(model_sigmas: Sequence[float], steps: int) -> list[float]:
    """Reproduce ComfyUI's simple scheduler against a sigma table."""

    sigmas = as_float_list(model_sigmas)
    validate_steps(steps)
    if not sigmas:
        raise ValueError("model_sigmas must not be empty")

    if steps > len(sigmas) - 1:
        out = [
            _sigma_at_position(sigmas, 1.0 - step / steps, "linear")
            for step in range(steps)
        ]
        out.append(0.0)
        return out

    stride = len(sigmas) / steps
    out = []
    for step in range(steps):
        index = -(1 + int(step * stride))
        out.append(float(sigmas[index]))
    out.append(0.0)
    return out


def build_early_dense_sigmas(
    model_sigmas: Sequence[float],
    steps: int,
    *,
    early_step_ratio: float = 0.50,
    mid_step_ratio: float = 0.32,
    early_end: float = 0.70,
    mid_end: float = 0.22,
    interpolation: str = "linear",
) -> list[float]:
    """Build a simple-derived schedule with extra high-noise transitions."""

    sigmas = as_float_list(model_sigmas)
    validate_steps(steps)
    _validate_boundaries(early_end, mid_end)
    if len(sigmas) < 2:
        raise ValueError("model_sigmas must contain at least two entries")
    if interpolation not in {"linear", "nearest"}:
        raise ValueError("interpolation must be 'linear' or 'nearest'")

    phase_steps = allocate_phase_steps(
        steps,
        early_step_ratio=early_step_ratio,
        mid_step_ratio=mid_step_ratio,
    )
    positions = build_phase_positions(
        phase_steps,
        early_end=early_end,
        mid_end=mid_end,
    )
    return [_sigma_at_position(sigmas, pos, interpolation) for pos in positions]


def build_anchored_sigmas(
    model_sigmas: Sequence[float],
    anchor_positions: Sequence[float],
    interval_steps: Sequence[int],
    *,
    interpolation: str = "linear",
) -> list[float]:
    """Build sigmas from explicit descending normalized anchors."""

    sigmas = as_float_list(model_sigmas)
    anchors = as_float_list(anchor_positions)
    steps = list(interval_steps)

    if len(sigmas) < 2:
        raise ValueError("model_sigmas must contain at least two entries")
    if len(anchors) < 2:
        raise ValueError("anchor_positions must contain at least two entries")
    if len(steps) != len(anchors) - 1:
        raise ValueError("interval_steps must have one entry per anchor interval")
    if any(step < 1 for step in steps):
        raise ValueError("interval_steps entries must be at least 1")
    if anchors[0] != 1.0 or anchors[-1] != 0.0:
        raise ValueError("anchor_positions must start at 1.0 and end at 0.0")
    if any(left <= right for left, right in zip(anchors, anchors[1:])):
        raise ValueError("anchor_positions must be strictly descending")
    if interpolation not in {"linear", "nearest"}:
        raise ValueError("interpolation must be 'linear' or 'nearest'")

    positions = build_anchored_positions(anchors, steps)
    return [_sigma_at_position(sigmas, pos, interpolation) for pos in positions]


def build_anchored_positions(
    anchor_positions: Sequence[float],
    interval_steps: Sequence[int],
) -> list[float]:
    """Build descending positions while preserving all supplied anchors."""

    anchors = as_float_list(anchor_positions)
    steps = list(interval_steps)

    positions: list[float] = []
    for index, count in enumerate(steps):
        segment = linspace(anchors[index], anchors[index + 1], count)
        if index > 0:
            segment = segment[1:]
        positions.extend(segment)

    positions[-1] = 0.0
    return positions


def early_dense_simple_scheduler(
    model_sampling: object,
    steps: int,
    *,
    early_step_ratio: float = 0.50,
    mid_step_ratio: float = 0.32,
    early_end: float = 0.70,
    mid_end: float = 0.22,
    interpolation: str = "linear",
):
    """ComfyUI-style scheduler wrapper returning a torch FloatTensor."""

    if not hasattr(model_sampling, "sigmas"):
        raise ValueError("model_sampling must expose a sigmas attribute")

    values = build_early_dense_sigmas(
        model_sampling.sigmas,
        steps,
        early_step_ratio=early_step_ratio,
        mid_step_ratio=mid_step_ratio,
        early_end=early_end,
        mid_end=mid_end,
        interpolation=interpolation,
    )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for ComfyUI scheduler output") from exc

    source = model_sampling.sigmas
    device = getattr(source, "device", None)
    dtype = getattr(source, "dtype", None)
    if dtype is None:
        dtype = torch.float32
    return torch.tensor(values, dtype=dtype, device=device)


def allocate_phase_steps(
    steps: int,
    *,
    early_step_ratio: float = 0.50,
    mid_step_ratio: float = 0.32,
) -> PhaseSteps:
    """Allocate integer transition counts to early/mid/late phases."""

    validate_steps(steps)
    if steps < 3:
        raise ValueError("early-dense scheduling requires at least 3 steps")

    late_step_ratio = 1.0 - early_step_ratio - mid_step_ratio
    ratios = [early_step_ratio, mid_step_ratio, late_step_ratio]
    if any(r <= 0 for r in ratios):
        raise ValueError("phase step ratios must be positive and sum below 1")

    total_ratio = sum(ratios)
    raw = [steps * ratio / total_ratio for ratio in ratios]
    base = [max(1, floor(value)) for value in raw]

    while sum(base) > steps:
        index = max(range(3), key=lambda i: base[i])
        base[index] -= 1

    remainders = [value - floor(value) for value in raw]
    while sum(base) < steps:
        index = max(range(3), key=lambda i: remainders[i])
        base[index] += 1
        remainders[index] = 0.0

    return PhaseSteps(early=base[0], mid=base[1], late=base[2])


def build_phase_positions(
    phase_steps: PhaseSteps,
    *,
    early_end: float,
    mid_end: float,
) -> list[float]:
    """Build descending normalized sigma-table positions for all phases."""

    _validate_boundaries(early_end, mid_end)
    if phase_steps.total < 1:
        raise ValueError("phase_steps must contain at least one transition")

    positions = linspace(1.0, early_end, phase_steps.early)
    positions += linspace(early_end, mid_end, phase_steps.mid)[1:]
    positions += linspace(mid_end, 0.0, phase_steps.late)[1:]
    positions[-1] = 0.0
    return positions


def _sigma_at_position(
    sigmas: Sequence[float],
    position: float,
    interpolation: str,
) -> float:
    if position <= 0.0:
        return 0.0
    if position >= 1.0:
        return float(sigmas[-1])

    table_index = position * (len(sigmas) - 1)
    if interpolation == "nearest":
        return float(sigmas[round(table_index)])

    lower_index = floor(table_index)
    upper_index = min(lower_index + 1, len(sigmas) - 1)
    alpha = table_index - lower_index
    lower = float(sigmas[lower_index])
    upper = float(sigmas[upper_index])
    return lower + (upper - lower) * alpha


def _validate_boundaries(early_end: float, mid_end: float) -> None:
    if not (1.0 > early_end > mid_end > 0.0):
        raise ValueError("expected 1.0 > early_end > mid_end > 0.0")
