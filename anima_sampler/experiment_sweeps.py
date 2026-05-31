"""Parameter sweep parsing and matrix helpers for experiment nodes."""

from __future__ import annotations

import math
import re
from typing import Any

from .cfg_schedule import CFG_SCHEDULE_MODES
from .flow_constants import FLOW_SCHEDULES, FLOW_SOLVERS

PARAMETER_SWEEP_KEYS = [
    "seed",
    "steps",
    "flow_solver",
    "flow_er_order",
    "flow_pc3_gamma",
    "flow_pc3_tolerance",
    "flow_unipc_order",
    "flow_unipc_solver_type",
    "flow_unipc_lower_order_final",
    "flow_unipc_disable_corrector_first",
    "flow_unipc_thresholding",
    "flow_unipc_dynamic_thresholding_ratio",
    "flow_unipc_sample_max_value",
    "flow_schedule",
    "flow_shift",
    "flow_rho7_tail_auto",
    "final_clean_pass",
    "cfg_legacy_progress",
    "denoise_legacy_progress",
    "cosmos_sigma_max",
    "cosmos_sigma_min",
    "cfg",
    "cfg_schedule_mode",
    "cfg_early_scale",
    "cfg_early_ramp_end",
    "cfg_peak_boost",
    "cfg_bump_start",
    "cfg_bump_end",
    "cfg_beta_alpha",
    "cfg_beta_beta",
    "cfg_interval_start",
    "cfg_interval_rise_end",
    "cfg_interval_fall_start",
    "cfg_interval_end",
    "early_cfg_boost",
    "early_cfg_until",
    "rf_endpoint_noise_refresh_enabled",
    "rf_endpoint_noise_refresh_strength",
    "rf_endpoint_noise_refresh_until",
    "late_cfg_scale",
    "late_cfg_start",
]

NO_SECONDARY_SWEEP = "<none>"
PARAMETER_MATRIX_KEYS = [NO_SECONDARY_SWEEP, *PARAMETER_SWEEP_KEYS]

_INTEGER_PARAMETERS = {
    "seed",
    "steps",
    "flow_er_order",
    "flow_unipc_order",
    "flow_unipc_disable_corrector_first",
}
_BOOLEAN_PARAMETERS = {
    "cfg_legacy_progress",
    "denoise_legacy_progress",
    "flow_rho7_tail_auto",
    "final_clean_pass",
    "flow_unipc_lower_order_final",
    "flow_unipc_thresholding",
    "rf_endpoint_noise_refresh_enabled",
}
_ENUM_PARAMETERS = {
    "cfg_schedule_mode": CFG_SCHEDULE_MODES,
    "flow_solver": FLOW_SOLVERS,
    "flow_schedule": FLOW_SCHEDULES,
    "flow_unipc_solver_type": ["bh2", "bh1"],
}


def parse_sweep_values(text: str, parameter: str, *, max_runs: int = 8) -> list[int | float | str | bool]:
    """Parse comma/newline/space separated parameter values for a sweep."""

    if parameter not in PARAMETER_SWEEP_KEYS:
        raise ValueError(f"unsupported sweep_parameter: {parameter}")
    if max_runs < 1:
        raise ValueError("max_runs must be at least 1")

    parts = [part for part in re.split(r"[\s,;]+", text.strip()) if part]
    values: list[int | float | str | bool] = []
    for part in parts[:max_runs]:
        if parameter in _ENUM_PARAMETERS:
            if part not in _ENUM_PARAMETERS[parameter]:
                allowed = ", ".join(_ENUM_PARAMETERS[parameter])
                raise ValueError(f"{parameter} must be one of: {allowed}")
            values.append(part)
        elif parameter in _INTEGER_PARAMETERS:
            values.append(_parse_integer_value(part))
        elif parameter in _BOOLEAN_PARAMETERS:
            values.append(_parse_boolean_value(part))
        else:
            value = float(part)
            if math.isnan(value) or math.isinf(value):
                raise ValueError("sweep values must be finite numbers")
            values.append(value)
    return values


def build_parameter_combinations(
    primary_parameter: str,
    primary_values_text: str,
    secondary_parameter: str = NO_SECONDARY_SWEEP,
    secondary_values_text: str = "",
    *,
    max_runs: int = 18,
) -> list[dict[str, int | float | str | bool]]:
    """Build a deterministic sweep list, optionally as a two-parameter matrix."""

    if max_runs < 1:
        raise ValueError("max_runs must be at least 1")
    if primary_parameter not in PARAMETER_SWEEP_KEYS:
        raise ValueError(f"unsupported primary_sweep_parameter: {primary_parameter}")
    if secondary_parameter not in PARAMETER_MATRIX_KEYS:
        raise ValueError(f"unsupported secondary_sweep_parameter: {secondary_parameter}")
    if secondary_parameter == primary_parameter:
        raise ValueError("primary_sweep_parameter and secondary_sweep_parameter must differ")

    primary_values = parse_sweep_values(primary_values_text, primary_parameter, max_runs=max_runs)
    if not primary_values:
        raise ValueError("primary_sweep_values must contain at least one value")

    if secondary_parameter == NO_SECONDARY_SWEEP:
        return [{primary_parameter: value} for value in primary_values[:max_runs]]

    secondary_values = parse_sweep_values(
        secondary_values_text,
        secondary_parameter,
        max_runs=max_runs,
    )
    if not secondary_values:
        raise ValueError("secondary_sweep_values must contain at least one value")

    combinations: list[dict[str, int | float | str | bool]] = []
    for secondary_value in secondary_values:
        for primary_value in primary_values:
            combination = {
                primary_parameter: primary_value,
                secondary_parameter: secondary_value,
            }
            if not _is_allowed_schedule_solver_combination(combination):
                continue
            combinations.append(combination)
            if len(combinations) >= max_runs:
                return combinations
    return combinations


def _is_allowed_schedule_solver_combination(overrides: dict[str, Any]) -> bool:
    """Return whether a schedule/solver pair should be included in a matrix."""

    if "flow_schedule" not in overrides or "flow_solver" not in overrides:
        return True

    return True


def _parse_integer_value(text: str) -> int:
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)

    value = float(text)
    if math.isnan(value) or math.isinf(value):
        raise ValueError("sweep values must be finite numbers")
    if not value.is_integer():
        raise ValueError("integer sweep values must be whole numbers")
    return int(value)


def _parse_boolean_value(text: str) -> bool:
    value = text.strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise ValueError("boolean sweep values must be true/false, 1/0, yes/no, or on/off")
