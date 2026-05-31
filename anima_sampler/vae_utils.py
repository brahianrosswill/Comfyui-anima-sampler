"""Helpers for decoding sampler latents into ComfyUI image tensors."""

from __future__ import annotations


def _decode_latent_image(vae, latent: dict):
    if vae is None:
        return None
    if "samples" not in latent:
        raise ValueError("latent output must contain a 'samples' tensor")

    samples = latent["samples"]
    ndim = int(getattr(samples, "ndim", len(samples.shape)))
    latent_dim = _vae_latent_dim(vae)
    expected_ndim = latent_dim + 2 if latent_dim is not None else None

    if expected_ndim == 5:
        if ndim == 4:
            samples = samples.unsqueeze(2)
        elif ndim == 5:
            if int(samples.shape[2]) != 1:
                raise ValueError("VAE image output requires a single-frame latent")
        else:
            raise ValueError("VAE image output requires a 4D or 5D latent")
    elif expected_ndim == 4:
        if ndim == 5:
            if int(samples.shape[2]) != 1:
                raise ValueError("VAE image output requires a single-frame latent")
            samples = samples.squeeze(2)
        elif ndim != 4:
            raise ValueError("VAE image output requires a 4D or 5D latent")
    elif ndim not in {4, 5}:
        raise ValueError("VAE image output requires a 4D or 5D latent")

    return _normalize_decoded_image(vae.decode(samples))


def _vae_latent_dim(vae):
    latent_dim = getattr(vae, "latent_dim", None)
    if latent_dim is None:
        return None
    try:
        return int(latent_dim)
    except (TypeError, ValueError):
        return None


def _normalize_decoded_image(image):
    ndim = int(getattr(image, "ndim", len(image.shape)))
    if ndim == 5:
        if int(image.shape[1]) != 1:
            raise ValueError("VAE image output requires a single-frame image")
        image = image[:, 0]
        ndim = 4
    if ndim != 4:
        raise ValueError("VAE image output requires a 4D image tensor")
    return image
