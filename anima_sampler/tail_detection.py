"""Schedule-tail and trace-phase detection helpers."""

from __future__ import annotations

import math

from .flow_math import _scalar_float


def _sampler_trace_phase(step_index: int, total_steps: int, tail_start_step: int | None) -> str:
    if tail_start_step is not None and step_index >= tail_start_step:
        return "tail"
    progress = step_index / max(total_steps - 1, 1)
    if progress < 0.25:
        return "high"
    if progress >= 0.68:
        return "tail"
    return "body"


def _active_integration_steps(torch, sigmas, *, final_clean_pass: bool) -> int:
    total = int(sigmas.shape[0]) - 1
    if (
        bool(final_clean_pass)
        and total >= 1
        and _scalar_float(torch, sigmas[-1]) <= 0.0
    ):
        return max(1, total - 1)
    return total


def _accelerating_tail_start_step(torch, sigmas) -> int | None:
    finite_count = int(sigmas.shape[0]) - 1
    if finite_count < 8:
        return None

    times = [_scalar_float(torch, sigmas[index]) for index in range(finite_count)]
    ells = [-math.log(max(value, 1e-12)) for value in times]
    gaps = [right - left for left, right in zip(ells, ells[1:])]
    if len(gaps) < 4:
        return None

    for index in range(3, len(gaps)):
        t_value = min(max(times[index], 1e-12), 1.0 - 1e-12)
        lambda_value = math.log((1.0 - t_value) / t_value)
        if lambda_value < 0.0:
            continue

        history = gaps[max(0, index - 6) : index]
        if len(history) < 3:
            continue
        base_gap = sorted(history)[len(history) // 2]
        if base_gap <= 0.0:
            continue
        local_future = gaps[index : min(len(gaps), index + 3)]
        if gaps[index] >= max(1.65 * base_gap, base_gap + 0.04) and all(
            gap >= 1.25 * base_gap for gap in local_future
        ):
            return index
    return None


def _hybrid_tail_start_step(
    torch,
    sigmas,
    flow_schedule: str,
    *,
    flow_shift: float = 1.0,
    flow_rho7_tail_auto: bool = False,
) -> int | None:
    is_rho_tail = str(flow_schedule) == "flow_cosmos_rho7" and bool(flow_rho7_tail_auto)
    is_shift_tail = str(flow_schedule) == "flow_cosmos_rf_tail"
    is_s_tail = str(flow_schedule) == "flow_rf_linear_s_tail_shift5"
    if is_s_tail:
        return _accelerating_tail_start_step(torch, sigmas)
    if not (is_rho_tail or is_shift_tail):
        return None

    finite_count = int(sigmas.shape[0]) - 1
    if finite_count < 3:
        return None

    times = [_scalar_float(torch, sigmas[index]) for index in range(finite_count)]
    ells = [-math.log(max(value, 1e-12)) for value in times]
    gaps = [right - left for left, right in zip(ells, ells[1:])]
    if len(gaps) < 2:
        return None

    for index in range(len(gaps) - 1):
        t_value = min(max(times[index], 1e-12), 1.0 - 1e-12)
        sigma = t_value / (1.0 - t_value)
        lambda_value = -math.log(max(sigma, 1e-12))
        if lambda_value < -1e-6:
            continue

        tail_gaps = gaps[index:]
        average_gap = sum(tail_gaps) / len(tail_gaps)
        tolerance = max(1e-5, abs(average_gap) * 1e-4)
        if max(abs(gap - average_gap) for gap in tail_gaps) <= tolerance:
            return index
    return None
