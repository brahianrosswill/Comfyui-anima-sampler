"""Shared math helpers for sigma schedule builders."""

from __future__ import annotations

from math import exp, log
from typing import Iterable


def linspace(start: float, end: float, transitions: int) -> list[float]:
    if transitions < 1:
        raise ValueError("transitions must be at least 1")
    step = (end - start) / transitions
    return [start + step * index for index in range(transitions + 1)]


def logspace(start: float, end: float, count: int) -> list[float]:
    if count < 1:
        raise ValueError("count must be at least 1")
    if count == 1:
        return [float(start)]

    log_start = log(start)
    log_end = log(end)
    step = (log_end - log_start) / (count - 1)
    return [exp(log_start + step * index) for index in range(count)]


def rho_space_descending(
    start: float,
    end: float,
    count: int,
    *,
    order: float,
) -> list[float]:
    if count < 1:
        raise ValueError("count must be at least 1")
    if count == 1:
        return [float(start)]

    start_root = start ** (1.0 / order)
    end_root = end ** (1.0 / order)
    step = (end_root - start_root) / (count - 1)
    return [(start_root + step * index) ** order for index in range(count)]


def ell_space_descending(start: float, end: float, count: int) -> list[float]:
    if count < 1:
        raise ValueError("count must be at least 1")
    if count == 1:
        return [float(start)]

    ell_start = ell_from_external_sigma(start)
    ell_end = ell_from_external_sigma(end)
    step = (ell_end - ell_start) / (count - 1)
    return [external_sigma_from_ell(ell_start + step * index) for index in range(count)]


def ell_from_external_sigma(sigma: float) -> float:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return log(1.0 + 1.0 / float(sigma))


def external_sigma_from_ell(ell: float) -> float:
    return 1.0 / (exp(float(ell)) - 1.0)


def as_float_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def validate_steps(steps: int) -> None:
    if not isinstance(steps, int):
        raise TypeError("steps must be an int")
    if steps < 1:
        raise ValueError("steps must be at least 1")
