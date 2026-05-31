"""Cosmos and Predict2-style normalized flow schedule builders."""

from __future__ import annotations

from bisect import bisect_left
from math import exp, log
from typing import Sequence

from .schedule_math import (
    ell_from_external_sigma,
    ell_space_descending,
    linspace,
    logspace,
    rho_space_descending,
    validate_steps,
)


def build_flow_cosmos_sigmas(
    steps: int,
    *,
    sigma_max: float = 80.0,
    sigma_min: float = 0.002,
) -> list[float]:
    """Build a Cosmos RFlow-shaped schedule in normalized Flow time."""

    validate_steps(steps)
    if not (0.0 < sigma_min < sigma_max):
        raise ValueError("expected 0 < sigma_min < sigma_max")

    external_sigmas = logspace(sigma_max, sigma_min, steps)
    return [
        _cosmos_rflow_time(sigma, sigma_max=sigma_max)
        for sigma in external_sigmas
    ] + [0.0]


def build_flow_cosmos_beta_sigmas(
    steps: int,
    *,
    beta: float = 0.0,
    sigma_max: float = 80.0,
    sigma_min: float = 0.002,
) -> list[float]:
    """Build a report-aligned beta-shifted Cosmos Flow schedule."""

    validate_steps(steps)
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if not (0.0 < sigma_min < sigma_max):
        raise ValueError("expected 0 < sigma_min < sigma_max")
    if beta == 0.0:
        return build_flow_cosmos_sigmas(
            steps,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
        )

    shifted_sigma_max = sigma_max * beta
    external_sigmas = logspace(shifted_sigma_max, sigma_min * beta, steps)
    return [
        _cosmos_rflow_time(sigma, sigma_max=shifted_sigma_max)
        for sigma in external_sigmas
    ] + [0.0]


def build_flow_cosmos_shift_rf_tail_sigmas(
    steps: int,
    *,
    beta: float,
    tail_delta_ell_max: float = 0.5,
    sigma_max: float = 80.0,
    sigma_min: float = 0.002,
    sigma_start: float | None = None,
) -> list[float]:
    """Build a beta-shifted high/mid schedule with an RF-native tail."""

    validate_steps(steps)
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if tail_delta_ell_max <= 0.0:
        raise ValueError("tail_delta_ell_max must be positive")
    if not (0.0 < sigma_min < sigma_max):
        raise ValueError("expected 0 < sigma_min < sigma_max")

    shifted_sigma_max = sigma_max * beta
    actual_sigma_start = shifted_sigma_max if sigma_start is None else float(sigma_start)
    if actual_sigma_start <= sigma_min:
        raise ValueError("expected sigma_start > sigma_min")

    shifted_sigma_min = sigma_min * beta
    if steps < 3 or actual_sigma_start <= shifted_sigma_min:
        external_sigmas = logspace(actual_sigma_start, sigma_min, steps)
    else:
        reference_shifted = logspace(actual_sigma_start, shifted_sigma_min, steps)
        external_sigmas = _rho_with_auto_ell_tail(
            reference_shifted,
            sigma_min=sigma_min,
            max_delta_ell=float(tail_delta_ell_max),
        )

    time_sigma_max = max(actual_sigma_start, shifted_sigma_max)
    return [
        _cosmos_rflow_time(sigma, sigma_max=time_sigma_max)
        for sigma in external_sigmas
    ] + [0.0]


def build_flow_cosmos_lambda_biased_sigmas(
    steps: int,
    *,
    strength: str = "default",
    sigma_max: float = 80.0,
    sigma_min: float = 0.002,
    grid_size: int = 20000,
) -> list[float]:
    """Build a lambda-density-shaped Cosmos Flow schedule."""

    validate_steps(steps)
    if strength not in _LAMBDA_BIASED_PROFILES:
        allowed = ", ".join(sorted(_LAMBDA_BIASED_PROFILES))
        raise ValueError(f"strength must be one of: {allowed}")
    if not (0.0 < sigma_min < sigma_max):
        raise ValueError("expected 0 < sigma_min < sigma_max")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    if steps == 1:
        return [_cosmos_rflow_time(sigma_max, sigma_max=sigma_max), 0.0]

    lambda_start = -log(sigma_max)
    lambda_end = -log(sigma_min)
    lambda_grid = linspace(lambda_start, lambda_end, grid_size - 1)
    profile = _LAMBDA_BIASED_PROFILES[strength]
    density = [_lambda_biased_density(value, profile) for value in lambda_grid]

    cdf = [0.0]
    total = 0.0
    for index in range(1, len(lambda_grid)):
        delta = lambda_grid[index] - lambda_grid[index - 1]
        total += 0.5 * (density[index] + density[index - 1]) * delta
        cdf.append(total)
    if total <= 0.0:
        raise ValueError("lambda density integral must be positive")
    cdf = [value / total for value in cdf]

    out = []
    for step in range(steps):
        target = step / (steps - 1)
        lambda_value = _inverse_monotonic_grid(lambda_grid, cdf, target)
        sigma = exp(-lambda_value)
        out.append(_cosmos_rflow_time(sigma, sigma_max=sigma_max))
    out.append(0.0)
    return out


def build_flow_cosmos_rho_sigmas(
    steps: int,
    *,
    order: float = 7.0,
    sigma_max: float = 80.0,
    sigma_min: float = 0.002,
) -> list[float]:
    """Build a Predict2-style rho/order schedule in normalized RF time."""

    validate_steps(steps)
    if order <= 0.0:
        raise ValueError("order must be positive")
    if not (0.0 < sigma_min < sigma_max):
        raise ValueError("expected 0 < sigma_min < sigma_max")

    external_sigmas = rho_space_descending(
        sigma_max,
        sigma_min,
        steps,
        order=order,
    )
    return [_cosmos_rflow_time(sigma, sigma_max=sigma_max) for sigma in external_sigmas] + [0.0]


def build_flow_cosmos_rho_rf_tail_sigmas(
    steps: int,
    *,
    tail_lambda_start: float | None = 0.5,
    tail_delta_ell_max: float | None = None,
    order: float = 7.0,
    sigma_max: float = 80.0,
    sigma_min: float = 0.002,
) -> list[float]:
    """Build a rho7 high/mid prefix with an RF-native uniform-ell tail."""

    validate_steps(steps)
    if order <= 0.0:
        raise ValueError("order must be positive")
    if not (0.0 < sigma_min < sigma_max):
        raise ValueError("expected 0 < sigma_min < sigma_max")
    if tail_delta_ell_max is not None and tail_delta_ell_max <= 0.0:
        raise ValueError("tail_delta_ell_max must be positive")
    if tail_lambda_start is None and tail_delta_ell_max is None:
        raise ValueError("expected tail_lambda_start or tail_delta_ell_max")

    reference_rho = rho_space_descending(
        sigma_max,
        sigma_min,
        steps,
        order=order,
    )

    if steps < 3:
        external_sigmas = reference_rho
    elif tail_delta_ell_max is not None:
        external_sigmas = _rho_with_auto_ell_tail(
            reference_rho,
            sigma_min=sigma_min,
            max_delta_ell=float(tail_delta_ell_max),
        )
    else:
        sigma_switch = exp(-float(tail_lambda_start))
        external_sigmas = _rho_with_fixed_lambda_tail(
            reference_rho,
            sigma_switch=sigma_switch,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

    return [_cosmos_rflow_time(sigma, sigma_max=sigma_max) for sigma in external_sigmas] + [0.0]


def _rho_with_fixed_lambda_tail(
    reference_rho: Sequence[float],
    *,
    sigma_switch: float,
    sigma_min: float,
    sigma_max: float,
) -> list[float]:
    steps = len(reference_rho)
    if sigma_switch >= sigma_max:
        external_sigmas = ell_space_descending(sigma_max, sigma_min, steps)
    elif sigma_switch <= sigma_min or steps < 3:
        external_sigmas = list(reference_rho)
    else:
        prefix = [sigma for sigma in reference_rho if sigma > sigma_switch]
        prefix.append(sigma_switch)
        if len(prefix) > steps - 1:
            prefix = list(reference_rho[: steps - 1])

        tail_count = steps - len(prefix)
        if tail_count < 1:
            return list(reference_rho)
        tail = ell_space_descending(sigma_switch, sigma_min, tail_count + 1)[1:]
        external_sigmas = prefix + tail
    return external_sigmas


def _rho_with_auto_ell_tail(
    reference_rho: Sequence[float],
    *,
    sigma_min: float,
    max_delta_ell: float,
) -> list[float]:
    latest_valid_index: int | None = None
    ell_end = ell_from_external_sigma(sigma_min)

    for index in range(len(reference_rho) - 1):
        sigma = reference_rho[index]
        current_lambda = -log(sigma)
        if current_lambda <= 0.0:
            continue

        tail_count = len(reference_rho) - index - 1
        if tail_count < 1:
            continue

        tail_delta_ell = (ell_end - ell_from_external_sigma(sigma)) / tail_count
        if tail_delta_ell <= max_delta_ell:
            latest_valid_index = index

    if latest_valid_index is not None:
        prefix = list(reference_rho[: latest_valid_index + 1])
        tail_count = len(reference_rho) - len(prefix)
        tail = ell_space_descending(prefix[-1], sigma_min, tail_count + 1)[1:]
        return prefix + tail

    for index in range(len(reference_rho) - 1):
        sigma = reference_rho[index]
        if -log(sigma) > 0.0:
            prefix = list(reference_rho[: index + 1])
            tail_count = len(reference_rho) - len(prefix)
            tail = ell_space_descending(prefix[-1], sigma_min, tail_count + 1)[1:]
            return prefix + tail

    return list(reference_rho)


_LAMBDA_BIASED_PROFILES = {
    "light": (
        (0.00, -2.3, 0.7),
        (0.20, 0.8, 1.4),
        (0.08, 3.2, 0.9),
    ),
    "default": (
        (0.10, -2.3, 0.7),
        (0.25, 0.8, 1.4),
        (0.12, 3.2, 0.9),
    ),
    "strong": (
        (0.15, -2.3, 0.7),
        (0.35, 0.8, 1.4),
        (0.18, 3.2, 0.9),
    ),
}


def _lambda_biased_density(value: float, profile: Sequence[tuple[float, float, float]]) -> float:
    density = 1.0
    for amp, center, width in profile:
        if amp <= 0.0:
            continue
        density += amp * exp(-0.5 * ((value - center) / width) ** 2)
    return density


def _inverse_monotonic_grid(values: Sequence[float], cdf: Sequence[float], target: float) -> float:
    if target <= 0.0:
        return float(values[0])
    if target >= 1.0:
        return float(values[-1])

    index = bisect_left(cdf, target)
    if index <= 0:
        return float(values[0])
    if index >= len(cdf):
        return float(values[-1])

    cdf0 = cdf[index - 1]
    cdf1 = cdf[index]
    if cdf1 <= cdf0:
        return float(values[index])

    alpha = (target - cdf0) / (cdf1 - cdf0)
    return float(values[index - 1] + alpha * (values[index] - values[index - 1]))


def _cosmos_rflow_time(sigma: float, *, sigma_max: float) -> float:
    if sigma <= 0.0:
        return 0.0
    sigma = min(float(sigma), float(sigma_max))
    return sigma / (sigma + 1.0)
