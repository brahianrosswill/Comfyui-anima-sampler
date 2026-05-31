"""ComfyUI version compatibility and latent preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any

from .latent_utils import _latent_channel_count, _shape_text


@dataclass(frozen=True)
class PreparedComfyLatent:
    latent_in_shape: str
    latent_sample_shape: str
    latent_image: Any
    added_temporal_dim: bool
    channel_adapter: str


def _fix_empty_latent_channels_compat(
    comfy_sample: Any,
    model: Any,
    latent_image: Any,
    downscale_ratio_spacial: Any,
    downscale_ratio_temporal: Any,
) -> Any:
    fix_empty_latent_channels = comfy_sample.fix_empty_latent_channels
    try:
        parameters = inspect.signature(fix_empty_latent_channels).parameters
    except (TypeError, ValueError):
        try:
            return fix_empty_latent_channels(
                model,
                latent_image,
                downscale_ratio_spacial,
                downscale_ratio_temporal,
            )
        except TypeError as original_error:
            try:
                return fix_empty_latent_channels(model, latent_image, downscale_ratio_spacial)
            except TypeError:
                raise original_error

    positional_count = 0
    accepts_varargs = False
    for parameter in parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            accepts_varargs = True
        elif parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_count += 1

    if accepts_varargs or positional_count >= 4:
        return fix_empty_latent_channels(
            model,
            latent_image,
            downscale_ratio_spacial,
            downscale_ratio_temporal,
        )
    if "downscale_ratio_temporal" in parameters:
        return fix_empty_latent_channels(
            model,
            latent_image,
            downscale_ratio_spacial=downscale_ratio_spacial,
            downscale_ratio_temporal=downscale_ratio_temporal,
        )
    return fix_empty_latent_channels(model, latent_image, downscale_ratio_spacial)


def prepare_latent_image_for_comfy(
    *,
    comfy_sample: Any,
    model: Any,
    latent: dict[str, Any],
) -> PreparedComfyLatent:
    if "samples" not in latent:
        raise ValueError("latent input must contain a 'samples' tensor")

    latent_samples = latent["samples"]
    latent_image = _fix_empty_latent_channels_compat(
        comfy_sample,
        model,
        latent_samples,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    source_channels = _latent_channel_count(latent_samples)
    fixed_channels = _latent_channel_count(latent_image)
    channel_adapter = ""
    if source_channels != fixed_channels:
        channel_adapter = (
            f"fixed latent channels {source_channels}->{fixed_channels} "
            "with ComfyUI model latent_format"
        )
        print(f"[comfyui-anima-sampler] {channel_adapter}")

    added_temporal_dim = len(latent_samples.shape) == 4 and len(latent_image.shape) == 5
    return PreparedComfyLatent(
        latent_in_shape=_shape_text(latent_samples),
        latent_sample_shape=_shape_text(latent_image),
        latent_image=latent_image,
        added_temporal_dim=added_temporal_dim,
        channel_adapter=channel_adapter,
    )
