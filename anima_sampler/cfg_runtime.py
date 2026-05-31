"""Dynamic CFG runtime helpers for the sampling loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cfg_schedule import cfg_at_progress, cfg_schedule_position


@dataclass
class CfgGuiderController:
    guider: Any
    original_cfg: Any
    can_set: bool
    warned: bool = False

    @classmethod
    def from_model(cls, model: Any) -> "CfgGuiderController":
        guider = getattr(model, "inner_model", None)
        can_set = guider is not None and (
            hasattr(guider, "set_cfg") or hasattr(guider, "cfg")
        )
        return cls(
            guider=guider,
            original_cfg=getattr(guider, "cfg", None),
            can_set=can_set,
        )

    def set(self, value: float) -> None:
        if self.can_set:
            set_cfg(self.guider, value)
        elif not self.warned:
            print("[comfyui-anima-sampler] Dynamic CFG unavailable; using static CFG.")
            self.warned = True

    def restore(self) -> None:
        if self.can_set and self.original_cfg is not None:
            set_cfg(self.guider, self.original_cfg)


@dataclass(frozen=True)
class CfgScheduleSettings:
    base_cfg: float
    mode: str
    early_cfg_boost: float
    early_cfg_until: float
    late_cfg_scale: float
    late_cfg_start: float
    early_scale: float
    early_ramp_end: float
    peak_boost: float
    bump_start: float
    bump_end: float
    beta_alpha: float
    beta_beta: float
    interval_start: float
    interval_rise_end: float
    interval_fall_start: float
    interval_end: float

    def value_at(self, progress: float) -> float:
        return cfg_for_progress(
            progress,
            base_cfg=self.base_cfg,
            cfg_schedule_mode=self.mode,
            early_cfg_boost=self.early_cfg_boost,
            early_cfg_until=self.early_cfg_until,
            late_cfg_scale=self.late_cfg_scale,
            late_cfg_start=self.late_cfg_start,
            cfg_early_scale=self.early_scale,
            cfg_early_ramp_end=self.early_ramp_end,
            cfg_peak_boost=self.peak_boost,
            cfg_bump_start=self.bump_start,
            cfg_bump_end=self.bump_end,
            cfg_beta_alpha=self.beta_alpha,
            cfg_beta_beta=self.beta_beta,
            cfg_interval_start=self.interval_start,
            cfg_interval_rise_end=self.interval_rise_end,
            cfg_interval_fall_start=self.interval_fall_start,
            cfg_interval_end=self.interval_end,
        )


def build_cfg_schedule_settings(
    *,
    base_cfg: float,
    cfg_schedule_mode: str,
    early_cfg_boost: float,
    early_cfg_until: float,
    late_cfg_scale: float,
    late_cfg_start: float,
    cfg_early_scale: float,
    cfg_early_ramp_end: float,
    cfg_peak_boost: float,
    cfg_bump_start: float,
    cfg_bump_end: float,
    cfg_beta_alpha: float,
    cfg_beta_beta: float,
    cfg_interval_start: float,
    cfg_interval_rise_end: float,
    cfg_interval_fall_start: float,
    cfg_interval_end: float,
) -> CfgScheduleSettings:
    return CfgScheduleSettings(
        base_cfg=float(base_cfg),
        mode=str(cfg_schedule_mode),
        early_cfg_boost=float(early_cfg_boost),
        early_cfg_until=float(early_cfg_until),
        late_cfg_scale=float(late_cfg_scale),
        late_cfg_start=float(late_cfg_start),
        early_scale=float(cfg_early_scale),
        early_ramp_end=float(cfg_early_ramp_end),
        peak_boost=float(cfg_peak_boost),
        bump_start=float(cfg_bump_start),
        bump_end=float(cfg_bump_end),
        beta_alpha=float(cfg_beta_alpha),
        beta_beta=float(cfg_beta_beta),
        interval_start=float(cfg_interval_start),
        interval_rise_end=float(cfg_interval_rise_end),
        interval_fall_start=float(cfg_interval_fall_start),
        interval_end=float(cfg_interval_end),
    )


def set_cfg_for_step(
    torch,
    sigmas,
    step_index: int,
    *,
    domain: str,
    total_steps: int,
    cfg_settings: CfgScheduleSettings,
    cfg_controller: CfgGuiderController,
) -> float:
    cfg_position = cfg_schedule_position(
        torch,
        sigmas[step_index],
        sigmas,
        step_index,
        domain=domain,
        total_steps=total_steps,
    )
    cfg_value = cfg_settings.value_at(cfg_position)
    cfg_controller.set(cfg_value)
    return cfg_value


def cfg_for_progress(
    progress: float,
    *,
    base_cfg: float,
    cfg_schedule_mode: str,
    early_cfg_boost: float,
    early_cfg_until: float,
    late_cfg_scale: float,
    late_cfg_start: float,
    cfg_early_scale: float,
    cfg_early_ramp_end: float,
    cfg_peak_boost: float,
    cfg_bump_start: float,
    cfg_bump_end: float,
    cfg_beta_alpha: float,
    cfg_beta_beta: float,
    cfg_interval_start: float,
    cfg_interval_rise_end: float,
    cfg_interval_fall_start: float,
    cfg_interval_end: float,
) -> float:
    return cfg_at_progress(
        progress,
        base_cfg=base_cfg,
        cfg_schedule_mode=cfg_schedule_mode,
        early_cfg_boost=early_cfg_boost,
        early_cfg_until=early_cfg_until,
        late_cfg_scale=late_cfg_scale,
        late_cfg_start=late_cfg_start,
        cfg_early_scale=cfg_early_scale,
        cfg_early_ramp_end=cfg_early_ramp_end,
        cfg_peak_boost=cfg_peak_boost,
        cfg_bump_start=cfg_bump_start,
        cfg_bump_end=cfg_bump_end,
        cfg_beta_alpha=cfg_beta_alpha,
        cfg_beta_beta=cfg_beta_beta,
        cfg_interval_start=cfg_interval_start,
        cfg_interval_rise_end=cfg_interval_rise_end,
        cfg_interval_fall_start=cfg_interval_fall_start,
        cfg_interval_end=cfg_interval_end,
    )


def set_cfg(cfg_guider: Any, value: float) -> None:
    if hasattr(cfg_guider, "set_cfg"):
        cfg_guider.set_cfg(value)
    else:
        cfg_guider.cfg = value
