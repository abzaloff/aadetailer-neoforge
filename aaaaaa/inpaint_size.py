"""Align ADetailer's internal inpaint size to the active Forge VAE."""
from __future__ import annotations

from math import lcm
from numbers import Integral


def vae_spatial_alignment(model) -> tuple[int, int] | None:
    """Return (width, height) pixel multiples, or None for non-Forge models."""
    vae = getattr(getattr(model, "forge_objects", None), "vae", None)
    ratio = getattr(vae, "downscale_ratio", None)
    # Forge video VAEs use (temporal, height, width) ratios.
    if isinstance(ratio, (tuple, list)) and len(ratio) >= 2:
        height, width = ratio[-2:]
    else:
        width = height = ratio
    if all(isinstance(v, Integral) and not isinstance(v, bool) and v > 0 for v in (width, height)):
        return int(width), int(height)
    return None


def align_inpaint_size(
    width: int,
    height: int,
    model,
    *,
    image_size: tuple[int, int] | None = None,
    separate_size: bool = False,
    model_override: bool = False,
) -> tuple[int, int]:
    alignment = vae_spatial_alignment(model)
    if alignment is None:
        return width, height
    if model_override:
        # The replacement model/VAE is loaded later by process_images.
        # 16 covers the 8/16-pixel Forge VAEs without changing the source.
        alignment = tuple(lcm(v, 16) for v in alignment)

    def align(value: int, multiple: int, actual: int | None) -> int:
        if value > 0 and value % multiple == 0:
            return value
        if not separate_size and actual is not None:
            # Reuse the real generated size if Forge rounded this axis.
            # Do not replace a deliberately different Hires/skip-img2img size.
            if actual > 0 and actual % multiple == 0 and abs(actual - value) < multiple:
                return actual
        if separate_size:
            # Separate width/height settings are minimums for mask scaling.
            return max(multiple, ((value + multiple - 1) // multiple) * multiple)
        return max(multiple, ((value + multiple // 2) // multiple) * multiple)

    actual_w, actual_h = image_size or (None, None)
    return align(width, alignment[0], actual_w), align(height, alignment[1], actual_h)
