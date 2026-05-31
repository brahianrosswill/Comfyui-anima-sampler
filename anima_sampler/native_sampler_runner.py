"""Stock ComfyUI sampler runner used by comparison experiments."""

from __future__ import annotations

import importlib
from typing import Any

from .comfy_compat import prepare_latent_image_for_comfy
from .model_sampling_info import _describe_model_sampling_shift


def run_comfy_native_sampler(
    *,
    model: Any,
    positive: Any,
    negative: Any,
    latent: dict[str, Any],
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    sampler_name: str,
    scheduler: str,
    add_noise: bool,
    disable_pbar: bool,
) -> tuple[dict[str, Any], str]:
    """Run a stock ComfyUI sampler/scheduler pair for experiment baselines."""

    torch = importlib.import_module("torch")
    comfy_sample = importlib.import_module("comfy.sample")
    comfy_samplers = importlib.import_module("comfy.samplers")
    comfy_utils = importlib.import_module("comfy.utils")
    latent_preview = importlib.import_module("latent_preview")

    if hasattr(comfy_samplers, "SAMPLER_NAMES") and sampler_name not in comfy_samplers.SAMPLER_NAMES:
        raise ValueError(f"ComfyUI sampler is not available: {sampler_name}")
    if hasattr(comfy_samplers, "SCHEDULER_NAMES") and scheduler not in comfy_samplers.SCHEDULER_NAMES:
        raise ValueError(f"ComfyUI scheduler is not available: {scheduler}")

    prepared = prepare_latent_image_for_comfy(
        comfy_sample=comfy_sample,
        model=model,
        latent=latent,
    )
    latent_image = prepared.latent_image

    batch_index = latent.get("batch_index")
    if add_noise:
        noise = comfy_sample.prepare_noise(latent_image, seed, batch_index)
    else:
        noise = torch.zeros_like(latent_image)

    callback = latent_preview.prepare_callback(model, int(steps))
    progress_disabled = disable_pbar or not comfy_utils.PROGRESS_BAR_ENABLED
    noise_mask = latent.get("noise_mask")

    samples = comfy_sample.sample(
        model,
        noise,
        int(steps),
        float(cfg),
        str(sampler_name),
        str(scheduler),
        positive,
        negative,
        latent_image,
        denoise=float(denoise),
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=progress_disabled,
        seed=seed,
    )

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples

    log = "\n".join(
        [
            "ComfyNativeSampler",
            f"sampler: {sampler_name}",
            f"scheduler: {scheduler}",
            f"steps: {int(steps)}",
            "steps_semantics: ComfyUI native denoise steps",
            "estimated_model_calls: native sampler dependent",
            f"cfg: {float(cfg):.4f}",
            f"denoise: {float(denoise):.4f}",
            f"latent_in_shape: {prepared.latent_in_shape}",
            f"latent_sample_shape: {prepared.latent_sample_shape}",
            f"channel_adapter: {prepared.channel_adapter or 'none'}",
            f"model_sampling_shift: {_describe_model_sampling_shift(model, flow_schedule='native')}",
            f"add_noise: {bool(add_noise)}",
        ]
    )
    return out, log
