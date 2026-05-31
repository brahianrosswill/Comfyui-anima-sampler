"""Runtime support helpers for the Flow Matching sampling loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cfg_runtime import (
    CfgGuiderController,
    CfgScheduleSettings,
    build_cfg_schedule_settings,
    cfg_for_progress,
    set_cfg,
    set_cfg_for_step,
)
from .flow_math import _finite_schedule_terminal, _scalar_float
from .latent_utils import _restore_sampler_channels
from .sampler_trace import _append_sampler_trace, _record_model_call
from .solver_types import FlowERState, FlowPC3State, FlowUniPC2State
from .solvers.basic import rf_endpoint_noise_refresh


@dataclass(frozen=True)
class EndpointRefreshSettings:
    enabled: bool
    strength: float
    until: float


@dataclass(frozen=True)
class SamplerStepTraceContext:
    step_index: int
    total_steps: int
    solver: str
    phase: str
    t: Any
    t_next: Any
    cfg: float
    x_before: Any
    denoised: Any
    cache_used: bool
    cache_score: float | None
    model_calls_before: int | None


def model_current_denoised(
    model,
    x,
    t,
    s_in,
    *,
    denoise_mask,
    model_options,
    seed: int,
    stats: dict[str, Any] | None,
):
    denoised = model(
        x,
        t * s_in,
        denoise_mask=denoise_mask,
        model_options=model_options,
        seed=seed,
    )
    _record_model_call(stats)
    return _restore_sampler_channels(denoised, x)


def append_stat_value(
    torch,
    stats: dict[str, Any] | None,
    key: str,
    value,
) -> None:
    if stats is not None:
        stats.setdefault(key, []).append(_scalar_float(torch, value))


def record_pc3_correction(
    torch,
    stats: dict[str, Any] | None,
    phase: str,
    gamma,
) -> None:
    if stats is None:
        return
    stats["pc3_used_total"] = int(stats.get("pc3_used_total", 0)) + 1
    stats[f"pc3_used_{phase}"] = int(stats.get(f"pc3_used_{phase}", 0)) + 1
    append_stat_value(torch, stats, "gamma_pc3_values", gamma)


def reset_solver_states() -> tuple[FlowERState, FlowPC3State, FlowUniPC2State]:
    return FlowERState(), FlowPC3State(), FlowUniPC2State()


def apply_endpoint_refresh(
    torch,
    x_candidate,
    x_before,
    denoised,
    t,
    t_next,
    generator,
    settings: EndpointRefreshSettings,
):
    return rf_endpoint_noise_refresh(
        torch,
        x_candidate,
        x_before,
        denoised,
        t,
        t_next,
        generator,
        enabled=settings.enabled,
        refresh_strength=settings.strength,
        refresh_until=settings.until,
    )


def append_step_trace(
    torch,
    stats: dict[str, Any] | None,
    context: SamplerStepTraceContext,
    *,
    x_after,
    cfg_next=None,
    x_pred=None,
    x_corrected=None,
    endpoint_call: bool,
    predictor_order: int,
    corrector_order: int = 0,
    gamma=None,
    gamma3=None,
    refresh_applied: bool = False,
    note: str = "",
) -> None:
    _append_sampler_trace(
        torch,
        stats,
        step_index=context.step_index,
        total_steps=context.total_steps,
        solver=context.solver,
        phase=context.phase,
        t=context.t,
        t_next=context.t_next,
        cfg=context.cfg,
        cfg_next=cfg_next,
        x_before=context.x_before,
        x_after=x_after,
        denoised=context.denoised,
        x_pred=x_pred,
        x_corrected=x_corrected,
        cache_used=context.cache_used,
        cache_score=context.cache_score,
        endpoint_call=endpoint_call,
        predictor_order=predictor_order,
        corrector_order=corrector_order,
        gamma=gamma,
        gamma3=gamma3,
        refresh_applied=refresh_applied,
        model_calls_before=context.model_calls_before,
        note=note,
    )


def apply_final_clean_pass(
    torch,
    model,
    x,
    sigmas,
    *,
    total_steps: int,
    flow_solver: str,
    cfg_settings: CfgScheduleSettings,
    cfg_controller: CfgGuiderController,
    s_in,
    denoise_mask,
    model_options,
    seed: int,
    stats: dict[str, Any] | None,
):
    cfg_clean = cfg_settings.value_at(1.0)
    cfg_controller.set(cfg_clean)
    clean_sigma = (
        sigmas[total_steps]
        if total_steps < int(sigmas.shape[0])
        else _finite_schedule_terminal(torch, sigmas)
    )
    clean_x_before = x
    clean_calls_before = int(stats.get("model_calls", 0)) if stats is not None else None
    x = model_current_denoised(
        model,
        x,
        clean_sigma,
        s_in,
        denoise_mask=denoise_mask,
        model_options=model_options,
        seed=seed,
        stats=stats,
    )
    clean_context = SamplerStepTraceContext(
        step_index=total_steps,
        total_steps=total_steps,
        solver=flow_solver,
        phase="clean",
        t=clean_sigma,
        t_next=clean_sigma,
        cfg=cfg_clean,
        x_before=clean_x_before,
        denoised=x,
        cache_used=False,
        cache_score=None,
        model_calls_before=clean_calls_before,
    )
    append_step_trace(
        torch,
        stats,
        clean_context,
        x_after=x,
        x_pred=x,
        endpoint_call=True,
        predictor_order=0,
        corrector_order=0,
        note="final_clean_pass",
    )
    return x


def make_generator(torch, device, seed: int):
    try:
        generator = torch.Generator(device=device)
    except Exception:
        try:
            generator = torch.Generator(device="cpu")
        except Exception:
            return None

    try:
        generator.manual_seed(seed)
    except Exception:
        return None
    return generator
