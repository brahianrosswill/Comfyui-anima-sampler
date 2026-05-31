"""Shared parameter defaults and normalization for ComfyUI nodes."""

from __future__ import annotations

import math

from .cfg_schedule import CFG_SCHEDULE_MODES
from .flow_constants import FLOW_SCHEDULES, FLOW_SOLVERS

DEFAULT_FLOW_SCHEDULE = "flow_rf_linear_shift"
DEFAULT_PUBLIC_CFG_MODE = "const"
PUBLIC_CFG_MODES = ["const", "bump cfg", "ramp cfg"]
NO_FINAL_CLEAN_DISCONNECTED_SCHEDULES = {
    "flow_rf_linear_shift",
    "flow_rf_linear_s_tail_shift5",
}

ANIMA_FLOW_BASELINE = {
    "steps": 35,
    "cfg": 7.0,
    "flow_solver": "flow_unipc2_x0",
    "flow_er_order": 2,
    "flow_pc3_gamma": 1.0,
    "flow_pc3_tolerance": 0.005,
    "flow_unipc_order": 2,
    "flow_unipc_solver_type": "bh2",
    "flow_unipc_lower_order_final": True,
    "flow_unipc_disable_corrector_first": 0,
    "flow_unipc_thresholding": False,
    "flow_unipc_dynamic_thresholding_ratio": 0.995,
    "flow_unipc_sample_max_value": 1.0,
    "flow_schedule": DEFAULT_FLOW_SCHEDULE,
    "flow_shift": 5.0,
    "flow_rho7_tail_auto": False,
    "final_clean_pass": False,
    "cosmos_sigma_max": 80.0,
    "cosmos_sigma_min": 0.002,
    "denoise_legacy_progress": False,
    "cfg_legacy_progress": False,
    "cfg_schedule_mode": "constant",
    "early_cfg_boost": 0.5,
    "early_cfg_until": 0.30,
    "late_cfg_scale": 1.0,
    "late_cfg_start": 0.76,
    "cfg_early_scale": 1.0,
    "cfg_early_ramp_end": 0.0,
    "cfg_peak_boost": 0.60,
    "cfg_bump_start": 0.0,
    "cfg_bump_end": 0.27,
    "cfg_beta_alpha": 2.0,
    "cfg_beta_beta": 3.0,
    "cfg_interval_start": 0.12,
    "cfg_interval_rise_end": 0.24,
    "cfg_interval_fall_start": 0.36,
    "cfg_interval_end": 0.58,
    "rf_endpoint_noise_refresh_enabled": False,
    "rf_endpoint_noise_refresh_strength": 0.15,
    "rf_endpoint_noise_refresh_until": 0.20,
}


def _normalize_settings_object(flow_settings) -> dict:
    if flow_settings is None:
        return _normalize_flow_params(ANIMA_FLOW_BASELINE)
    if not isinstance(flow_settings, dict):
        raise ValueError("flow_settings must be an Anima Flow Settings object")

    out = dict(ANIMA_FLOW_BASELINE)
    for key, value in flow_settings.items():
        if key in ANIMA_FLOW_BASELINE:
            out[key] = value
    return _normalize_flow_params(out)


def _apply_disconnected_sampler_defaults(params: dict, flow_settings) -> dict:
    out = dict(params)
    if flow_settings is None:
        out["final_clean_pass"] = (
            out["flow_schedule"] not in NO_FINAL_CLEAN_DISCONNECTED_SCHEDULES
        )
    return _normalize_flow_params(out)


def _constant_linear_shift_profile(*, seed, steps, cfg, flow_solver: str) -> dict:
    params = _normalize_flow_params(
        {
            **ANIMA_FLOW_BASELINE,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "flow_solver": flow_solver,
            "flow_schedule": "flow_rf_linear_shift",
            "flow_shift": 5.0,
            "final_clean_pass": False,
        }
    )
    return _apply_public_cfg_mode(params, "const")


def _apply_public_cfg_mode(params: dict, cfg_mode: str) -> dict:
    out = dict(params)
    mode = str(cfg_mode).strip().lower()
    if mode == "bump cfg":
        out["cfg_schedule_mode"] = "beta_bump"
    elif mode == "ramp cfg":
        out["cfg_schedule_mode"] = "low_to_high"
        out["cfg_early_scale"] = min(1.0, 4.5 / max(float(out["cfg"]), 1e-6))
        out["cfg_early_ramp_end"] = 0.0
        out["cfg_peak_boost"] = 0.0
        out["cfg_interval_start"] = 0.24
        out["cfg_interval_rise_end"] = 0.66
        out["cfg_interval_fall_start"] = 1.0
        out["cfg_interval_end"] = 1.0
        out["late_cfg_scale"] = 1.0
    elif mode == "const":
        out["cfg_schedule_mode"] = "constant"
        out["cfg_early_scale"] = 1.0
        out["cfg_early_ramp_end"] = 0.0
        out["cfg_peak_boost"] = 0.0
        out["late_cfg_scale"] = 1.0
    else:
        allowed = ", ".join(PUBLIC_CFG_MODES)
        raise ValueError(f"cfg_mode must be one of: {allowed}")
    return _normalize_flow_params(out)


def _normalize_flow_params(params: dict) -> dict:
    out = dict(params)
    out["steps"] = int(out["steps"])
    out["cfg"] = float(out["cfg"])
    out["flow_solver"] = str(out["flow_solver"])
    out["flow_er_order"] = int(out["flow_er_order"])
    out["flow_pc3_gamma"] = float(out["flow_pc3_gamma"])
    out["flow_pc3_tolerance"] = float(out["flow_pc3_tolerance"])
    out["flow_unipc_order"] = int(out["flow_unipc_order"])
    out["flow_unipc_solver_type"] = str(out["flow_unipc_solver_type"])
    out["flow_unipc_lower_order_final"] = _as_bool(out["flow_unipc_lower_order_final"])
    out["flow_unipc_disable_corrector_first"] = int(out["flow_unipc_disable_corrector_first"])
    out["flow_unipc_thresholding"] = _as_bool(out["flow_unipc_thresholding"])
    out["flow_unipc_dynamic_thresholding_ratio"] = float(out["flow_unipc_dynamic_thresholding_ratio"])
    out["flow_unipc_sample_max_value"] = float(out["flow_unipc_sample_max_value"])
    out["flow_schedule"] = str(out["flow_schedule"])
    if out["flow_schedule"] == "flow_cosmos_rho7_rf_tail_auto":
        out["flow_schedule"] = "flow_cosmos_rho7"
        out["flow_rho7_tail_auto"] = True
    out["flow_shift"] = float(out["flow_shift"])
    out["flow_rho7_tail_auto"] = _as_bool(out["flow_rho7_tail_auto"])
    out["final_clean_pass"] = _as_bool(out["final_clean_pass"])
    out["cosmos_sigma_max"] = float(out["cosmos_sigma_max"])
    out["cosmos_sigma_min"] = float(out["cosmos_sigma_min"])
    out["denoise_legacy_progress"] = _as_bool(out["denoise_legacy_progress"])
    out["cfg_legacy_progress"] = _as_bool(out["cfg_legacy_progress"])
    out["cfg_schedule_mode"] = str(out["cfg_schedule_mode"])
    out["cfg_early_scale"] = float(out["cfg_early_scale"])
    out["cfg_early_ramp_end"] = float(out["cfg_early_ramp_end"])
    out["cfg_peak_boost"] = float(out["cfg_peak_boost"])
    out["cfg_bump_start"] = float(out["cfg_bump_start"])
    out["cfg_bump_end"] = float(out["cfg_bump_end"])
    out["cfg_beta_alpha"] = float(out["cfg_beta_alpha"])
    out["cfg_beta_beta"] = float(out["cfg_beta_beta"])
    out["cfg_interval_start"] = float(out["cfg_interval_start"])
    out["cfg_interval_rise_end"] = float(out["cfg_interval_rise_end"])
    out["cfg_interval_fall_start"] = float(out["cfg_interval_fall_start"])
    out["cfg_interval_end"] = float(out["cfg_interval_end"])
    out["early_cfg_boost"] = float(out["early_cfg_boost"])
    out["early_cfg_until"] = float(out["early_cfg_until"])
    out["late_cfg_scale"] = float(out["late_cfg_scale"])
    out["late_cfg_start"] = float(out["late_cfg_start"])
    out["rf_endpoint_noise_refresh_enabled"] = _as_bool(
        out["rf_endpoint_noise_refresh_enabled"]
    )
    out["rf_endpoint_noise_refresh_strength"] = float(
        out["rf_endpoint_noise_refresh_strength"]
    )
    out["rf_endpoint_noise_refresh_until"] = float(out["rf_endpoint_noise_refresh_until"])

    if out["flow_solver"] not in FLOW_SOLVERS:
        raise ValueError(f"unsupported flow_solver: {out['flow_solver']}")
    if not (1 <= out["flow_er_order"] <= 3):
        raise ValueError("flow_er_order must be in the range [1, 3]")
    if not (0.0 <= out["flow_pc3_gamma"] <= 1.0):
        raise ValueError("flow_pc3_gamma must be in the range [0, 1]")
    if not (0.0 < out["flow_pc3_tolerance"] <= 1.0):
        raise ValueError("flow_pc3_tolerance must be in the range (0, 1]")
    if not (1 <= out["flow_unipc_order"] <= 6):
        raise ValueError("flow_unipc_order must be in the range [1, 6]")
    if out["flow_unipc_solver_type"] not in {"bh1", "bh2"}:
        raise ValueError("flow_unipc_solver_type must be bh1 or bh2")
    if not (0 <= out["flow_unipc_disable_corrector_first"] <= 10):
        raise ValueError("flow_unipc_disable_corrector_first must be in the range [0, 10]")
    if not (0.0 < out["flow_unipc_dynamic_thresholding_ratio"] <= 1.0):
        raise ValueError("flow_unipc_dynamic_thresholding_ratio must be in the range (0, 1]")
    if out["flow_unipc_sample_max_value"] < 1.0:
        raise ValueError("flow_unipc_sample_max_value must be >= 1")
    if out["flow_schedule"] not in FLOW_SCHEDULES:
        raise ValueError(f"unsupported flow_schedule: {out['flow_schedule']}")
    if not math.isfinite(out["flow_shift"]) or out["flow_shift"] < 1.0:
        raise ValueError("flow_shift must be finite and >= 1")
    if not (
        math.isfinite(out["cosmos_sigma_min"])
        and math.isfinite(out["cosmos_sigma_max"])
        and 0.0 < out["cosmos_sigma_min"] < out["cosmos_sigma_max"]
    ):
        raise ValueError("expected finite 0 < cosmos_sigma_min < cosmos_sigma_max")
    if out["cfg_schedule_mode"] not in CFG_SCHEDULE_MODES:
        allowed = ", ".join(CFG_SCHEDULE_MODES)
        raise ValueError(f"cfg_schedule_mode must be one of: {allowed}")
    if not (0.0 <= out["cfg_early_scale"] <= 2.0):
        raise ValueError("cfg_early_scale must be in the range [0, 2]")
    if not (0.0 <= out["cfg_early_ramp_end"] <= 1.0):
        raise ValueError("cfg_early_ramp_end must be in the range [0, 1]")
    if out["cfg_peak_boost"] < 0.0:
        raise ValueError("cfg_peak_boost must be non-negative")
    if not (0.0 <= out["cfg_bump_start"] < out["cfg_bump_end"] <= 1.0):
        raise ValueError("expected 0 <= cfg_bump_start < cfg_bump_end <= 1")
    if out["cfg_beta_alpha"] <= 1.0 or out["cfg_beta_beta"] <= 1.0:
        raise ValueError("cfg_beta_alpha and cfg_beta_beta must be > 1")
    if not (
        0.0
        <= out["cfg_interval_start"]
        <= out["cfg_interval_rise_end"]
        <= out["cfg_interval_fall_start"]
        <= out["cfg_interval_end"]
        <= 1.0
    ):
        raise ValueError(
            "expected cfg_interval_start <= cfg_interval_rise_end <= "
            "cfg_interval_fall_start <= cfg_interval_end within [0, 1]"
        )
    if not (0.0 <= out["rf_endpoint_noise_refresh_strength"] <= 1.0):
        raise ValueError("rf_endpoint_noise_refresh_strength must be in the range [0, 1]")
    if not (0.0 <= out["rf_endpoint_noise_refresh_until"] <= 1.0):
        raise ValueError("rf_endpoint_noise_refresh_until must be in the range [0, 1]")
    return out


def _cfg_domain_from_settings(settings: dict) -> str:
    return "progress" if bool(settings["cfg_legacy_progress"]) else "lambda"


def _denoise_domain_from_settings(settings: dict) -> str:
    return "progress" if bool(settings["denoise_legacy_progress"]) else "lambda"


def _estimated_model_calls(settings: dict) -> int:
    steps = max(1, int(settings["steps"]))
    if settings["flow_solver"] in {
        "flow_heun",
        "flow_pc3_damped",
    }:
        calls = max(1, steps * 2 - 1)
        return calls + int(bool(settings["final_clean_pass"]))
    return steps + int(bool(settings["final_clean_pass"]))


def _format_settings_summary(settings: dict) -> str:
    return "\n".join(
        [
            "AnimaFlowSettings",
            (
                "note: optional advanced settings; sampler steps/cfg/cfg_mode/"
                "solver/scheduler/shift override this object"
            ),
            f"steps_default: {settings['steps']}",
            f"cfg_default: {settings['cfg']:.2f}",
            f"estimated_model_calls_at_defaults: {_estimated_model_calls(settings)}",
            f"sampler_default_flow_solver: {settings['flow_solver']}",
            f"sampler_default_flow_schedule: {settings['flow_schedule']}",
            f"sampler_default_flow_shift: {settings['flow_shift']:.4f}",
            f"flow_rho7_tail_auto: {settings['flow_rho7_tail_auto']}",
            f"final_clean_pass: {settings['final_clean_pass']}",
            f"flow_er_order: {settings['flow_er_order']}",
            f"flow_pc3_gamma: {settings['flow_pc3_gamma']:.4f}",
            f"flow_pc3_tolerance: {settings['flow_pc3_tolerance']:.6f}",
            f"flow_unipc_order: {settings['flow_unipc_order']}",
            f"flow_unipc_solver_type: {settings['flow_unipc_solver_type']}",
            f"flow_unipc_lower_order_final: {settings['flow_unipc_lower_order_final']}",
            (
                "flow_unipc_disable_corrector_first: "
                f"{settings['flow_unipc_disable_corrector_first']}"
            ),
            f"flow_unipc_thresholding: {settings['flow_unipc_thresholding']}",
            (
                "flow_unipc_dynamic_thresholding_ratio: "
                f"{settings['flow_unipc_dynamic_thresholding_ratio']:.4f}"
            ),
            f"flow_unipc_sample_max_value: {settings['flow_unipc_sample_max_value']:.4f}",
            f"cosmos_sigma_max: {settings['cosmos_sigma_max']:.4f}",
            f"cosmos_sigma_min: {settings['cosmos_sigma_min']:.6f}",
            f"cfg_schedule_mode: {settings['cfg_schedule_mode']}",
            f"cfg_schedule_domain: {_cfg_domain_from_settings(settings)}",
            f"cfg_early_scale: {settings['cfg_early_scale']:.4f}",
            f"cfg_early_ramp_end: {settings['cfg_early_ramp_end']:.4f}",
            f"cfg_peak_boost: {settings['cfg_peak_boost']:.4f}",
            f"cfg_bump_start: {settings['cfg_bump_start']:.4f}",
            f"cfg_bump_end: {settings['cfg_bump_end']:.4f}",
            f"cfg_beta_alpha: {settings['cfg_beta_alpha']:.4f}",
            f"cfg_beta_beta: {settings['cfg_beta_beta']:.4f}",
            f"late_cfg_scale: {settings['late_cfg_scale']:.4f}",
            f"late_cfg_start: {settings['late_cfg_start']:.4f}",
            f"cfg_legacy_progress: {settings['cfg_legacy_progress']}",
            f"denoise_legacy_progress: {settings['denoise_legacy_progress']}",
            f"denoise_schedule_domain: {_denoise_domain_from_settings(settings)}",
            (
                "rf_endpoint_noise_refresh_enabled: "
                f"{settings['rf_endpoint_noise_refresh_enabled']}"
            ),
            (
                "rf_endpoint_noise_refresh_strength: "
                f"{settings['rf_endpoint_noise_refresh_strength']:.4f}"
            ),
            (
                "rf_endpoint_noise_refresh_until: "
                f"{settings['rf_endpoint_noise_refresh_until']:.4f}"
            ),
        ]
    )


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"expected boolean value, got: {value!r}")
