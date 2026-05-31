"""Primary ComfyUI sampler node implementation."""

from __future__ import annotations

from .flow_constants import FLOW_SCHEDULES, FLOW_SOLVERS
from .node_metadata import ANIMA_FLOW_SETTINGS, NODE_CATEGORY
from .node_runner import _run_sampler_with_params
from .node_settings import (
    ANIMA_FLOW_BASELINE,
    DEFAULT_PUBLIC_CFG_MODE,
    PUBLIC_CFG_MODES,
    _apply_disconnected_sampler_defaults,
    _apply_public_cfg_mode,
    _normalize_flow_params,
    _normalize_settings_object,
)
from .vae_utils import _decode_latent_image


class AnimaFlowCorrectiveSampler:
    """Standalone RC2 Flow sampler for Anima."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": ANIMA_FLOW_BASELINE["steps"],
                        "min": 1,
                        "max": 1000,
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg"],
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.1,
                    },
                ),
                "cfg_mode": (PUBLIC_CFG_MODES, {"default": DEFAULT_PUBLIC_CFG_MODE}),
                "flow_solver": (FLOW_SOLVERS, {"default": ANIMA_FLOW_BASELINE["flow_solver"]}),
                "flow_schedule": (FLOW_SCHEDULES, {"default": ANIMA_FLOW_BASELINE["flow_schedule"]}),
                "flow_shift": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["flow_shift"],
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.1,
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "add_noise": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "flow_settings": (ANIMA_FLOW_SETTINGS,),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("latent", "image", "log")
    FUNCTION = "sample"
    CATEGORY = NODE_CATEGORY

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        cfg_mode,
        flow_solver,
        flow_schedule,
        flow_shift,
        denoise,
        add_noise,
        flow_settings=None,
        vae=None,
    ):
        base_params = _normalize_settings_object(flow_settings)
        params = _normalize_flow_params(
            {
                **base_params,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "flow_solver": flow_solver,
                "flow_schedule": flow_schedule,
                "flow_shift": flow_shift,
            }
        )
        params = _apply_disconnected_sampler_defaults(params, flow_settings)
        params = _apply_public_cfg_mode(params, cfg_mode)
        latent_out, log = _run_sampler_with_params(
            model=model,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            params=params,
            denoise=denoise,
            add_noise=add_noise,
            disable_pbar=False,
        )
        image = _decode_latent_image(vae, latent_out)
        if vae is None:
            log = f"{log}\nimage_output: unavailable (connect VAE)"
        else:
            log = f"{log}\nimage_output: decoded with connected VAE"
        return latent_out, image, log
