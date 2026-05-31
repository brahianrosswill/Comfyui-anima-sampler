"""Advanced settings node for the Anima sampler."""

from __future__ import annotations

from .node_metadata import ANIMA_FLOW_SETTINGS, NODE_CATEGORY
from .node_settings import (
    ANIMA_FLOW_BASELINE,
    _format_settings_summary,
    _normalize_flow_params,
)


class AnimaFlowSettings:
    """Optional advanced settings for the standalone Anima Flow sampler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_er_order": (
                    "INT",
                    {
                        "default": ANIMA_FLOW_BASELINE["flow_er_order"],
                        "min": 1,
                        "max": 3,
                    },
                ),
                "flow_pc3_gamma": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["flow_pc3_gamma"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "flow_pc3_tolerance": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["flow_pc3_tolerance"],
                        "min": 0.0001,
                        "max": 0.05,
                        "step": 0.0005,
                    },
                ),
                "flow_unipc_order": (
                    "INT",
                    {
                        "default": ANIMA_FLOW_BASELINE["flow_unipc_order"],
                        "min": 1,
                        "max": 6,
                    },
                ),
                "flow_unipc_solver_type": (
                    ["bh2", "bh1"],
                    {"default": ANIMA_FLOW_BASELINE["flow_unipc_solver_type"]},
                ),
                "flow_unipc_lower_order_final": (
                    "BOOLEAN",
                    {"default": ANIMA_FLOW_BASELINE["flow_unipc_lower_order_final"]},
                ),
                "flow_unipc_disable_corrector_first": (
                    "INT",
                    {
                        "default": ANIMA_FLOW_BASELINE["flow_unipc_disable_corrector_first"],
                        "min": 0,
                        "max": 10,
                    },
                ),
                "flow_unipc_thresholding": (
                    "BOOLEAN",
                    {"default": ANIMA_FLOW_BASELINE["flow_unipc_thresholding"]},
                ),
                "flow_unipc_dynamic_thresholding_ratio": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["flow_unipc_dynamic_thresholding_ratio"],
                        "min": 0.5,
                        "max": 1.0,
                        "step": 0.001,
                    },
                ),
                "flow_unipc_sample_max_value": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["flow_unipc_sample_max_value"],
                        "min": 1.0,
                        "max": 10.0,
                        "step": 0.1,
                    },
                ),
                "cfg_early_scale": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_early_scale"],
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                    },
                ),
                "cfg_early_ramp_end": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_early_ramp_end"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "cfg_peak_boost": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_peak_boost"],
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.05,
                    },
                ),
                "cfg_bump_start": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_bump_start"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "cfg_bump_end": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_bump_end"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "cfg_beta_alpha": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_beta_alpha"],
                        "min": 1.0001,
                        "max": 20.0,
                        "step": 0.1,
                    },
                ),
                "cfg_beta_beta": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_beta_beta"],
                        "min": 1.0001,
                        "max": 20.0,
                        "step": 0.1,
                    },
                ),
                "late_cfg_scale": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["late_cfg_scale"],
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                    },
                ),
                "late_cfg_start": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["late_cfg_start"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "cfg_legacy_progress": (
                    "BOOLEAN",
                    {"default": ANIMA_FLOW_BASELINE["cfg_legacy_progress"]},
                ),
                "denoise_legacy_progress": (
                    "BOOLEAN",
                    {"default": ANIMA_FLOW_BASELINE["denoise_legacy_progress"]},
                ),
                "flow_rho7_tail_auto": (
                    "BOOLEAN",
                    {"default": ANIMA_FLOW_BASELINE["flow_rho7_tail_auto"]},
                ),
                "final_clean_pass": (
                    "BOOLEAN",
                    {"default": ANIMA_FLOW_BASELINE["final_clean_pass"]},
                ),
                "cosmos_sigma_max": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cosmos_sigma_max"],
                        "min": 1.0,
                        "max": 1000.0,
                        "step": 0.5,
                    },
                ),
                "cosmos_sigma_min": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cosmos_sigma_min"],
                        "min": 0.0001,
                        "max": 1.0,
                        "step": 0.0001,
                    },
                ),
                "rf_endpoint_noise_refresh_enabled": (
                    "BOOLEAN",
                    {"default": ANIMA_FLOW_BASELINE["rf_endpoint_noise_refresh_enabled"]},
                ),
                "rf_endpoint_noise_refresh_strength": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["rf_endpoint_noise_refresh_strength"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "rf_endpoint_noise_refresh_until": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["rf_endpoint_noise_refresh_until"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
            },
        }

    RETURN_TYPES = (ANIMA_FLOW_SETTINGS, "STRING")
    RETURN_NAMES = ("settings", "summary")
    FUNCTION = "build"
    CATEGORY = NODE_CATEGORY

    def build(
        self,
        flow_er_order,
        flow_pc3_gamma,
        flow_pc3_tolerance,
        flow_unipc_order,
        flow_unipc_solver_type,
        flow_unipc_lower_order_final,
        flow_unipc_disable_corrector_first,
        flow_unipc_thresholding,
        flow_unipc_dynamic_thresholding_ratio,
        flow_unipc_sample_max_value,
        cfg_early_scale,
        cfg_early_ramp_end,
        cfg_peak_boost,
        cfg_bump_start,
        cfg_bump_end,
        cfg_beta_alpha,
        cfg_beta_beta,
        late_cfg_scale,
        late_cfg_start,
        cfg_legacy_progress,
        denoise_legacy_progress,
        flow_rho7_tail_auto,
        final_clean_pass,
        cosmos_sigma_max,
        cosmos_sigma_min,
        rf_endpoint_noise_refresh_enabled,
        rf_endpoint_noise_refresh_strength,
        rf_endpoint_noise_refresh_until,
    ):
        settings = _normalize_flow_params(
            {
                **ANIMA_FLOW_BASELINE,
                "flow_er_order": flow_er_order,
                "flow_pc3_gamma": flow_pc3_gamma,
                "flow_pc3_tolerance": flow_pc3_tolerance,
                "flow_unipc_order": flow_unipc_order,
                "flow_unipc_solver_type": flow_unipc_solver_type,
                "flow_unipc_lower_order_final": flow_unipc_lower_order_final,
                "flow_unipc_disable_corrector_first": flow_unipc_disable_corrector_first,
                "flow_unipc_thresholding": flow_unipc_thresholding,
                "flow_unipc_dynamic_thresholding_ratio": flow_unipc_dynamic_thresholding_ratio,
                "flow_unipc_sample_max_value": flow_unipc_sample_max_value,
                "cfg_early_scale": cfg_early_scale,
                "cfg_early_ramp_end": cfg_early_ramp_end,
                "cfg_peak_boost": cfg_peak_boost,
                "cfg_bump_start": cfg_bump_start,
                "cfg_bump_end": cfg_bump_end,
                "cfg_beta_alpha": cfg_beta_alpha,
                "cfg_beta_beta": cfg_beta_beta,
                "late_cfg_scale": late_cfg_scale,
                "late_cfg_start": late_cfg_start,
                "cfg_legacy_progress": cfg_legacy_progress,
                "denoise_legacy_progress": denoise_legacy_progress,
                "flow_rho7_tail_auto": flow_rho7_tail_auto,
                "final_clean_pass": final_clean_pass,
                "cosmos_sigma_max": cosmos_sigma_max,
                "cosmos_sigma_min": cosmos_sigma_min,
                "rf_endpoint_noise_refresh_enabled": rf_endpoint_noise_refresh_enabled,
                "rf_endpoint_noise_refresh_strength": rf_endpoint_noise_refresh_strength,
                "rf_endpoint_noise_refresh_until": rf_endpoint_noise_refresh_until,
            }
        )
        return settings, _format_settings_summary(settings)
