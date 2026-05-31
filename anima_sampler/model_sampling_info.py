"""Best-effort descriptions of ComfyUI model_sampling behavior."""

from __future__ import annotations

from typing import Any


def _describe_model_sampling_shift(model: Any, *, flow_schedule: str) -> str:
    try:
        model_sampling = model.get_model_object("model_sampling")
    except Exception:
        return "unknown"

    class_name = type(model_sampling).__name__
    shift = getattr(model_sampling, "shift", None)
    if shift is not None:
        try:
            shift_text = f"{float(shift):.4g}"
        except (TypeError, ValueError):
            shift_text = str(shift)
        if class_name in {"ModelSamplingDiscreteFlow", "ModelSamplingFlux"}:
            rf_linear_schedules = {
                "flow_diffusers_linear_shift",
                "flow_rf_linear_shift",
                "flow_rf_linear_s_tail_shift5",
            }
            bypasses_native_shift = (
                flow_schedule.startswith("flow_cosmos")
                or flow_schedule in rf_linear_schedules
            )
            if bypasses_native_shift:
                return f"{class_name}: native shift={shift_text} present, bypassed by {flow_schedule}"
            return f"{class_name}: native shift={shift_text} baked into sigmas"
        return f"{class_name}: shift={shift_text}"

    if class_name == "ModelSamplingCosmosRFlow":
        return f"{class_name}: sigma-ratio flow table"
    return f"{class_name}: no shift attribute"
