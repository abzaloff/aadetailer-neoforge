"""Temporarily use normal inpainting rather than ANIMA reference editing."""

from __future__ import annotations

from contextlib import contextmanager

_KEY = "anima_do_reference"
_MISSING = object()


def is_anima_engine(model) -> bool:
    # Use the actual Forge engine, not checkpoint filenames or VAE type.
    return any(
        cls.__module__ == "backend.diffusion_engine.anima" and cls.__name__ == "Anima"
        for cls in type(model).__mro__
    )


@contextmanager
def suspend_anima_reference(opts, model, *, checkpoint_override=False):
    """Yield whether the setting was suspended; restore its exact prior state."""
    data = getattr(opts, "data", None)
    if not isinstance(data, dict) or not (
        is_anima_engine(model) or checkpoint_override
    ):
        yield False
        return

    # A checkpoint override is loaded inside process_images. In that case,
    # guard the ANIMA-only option even if the currently loaded engine differs.
    original = data.get(_KEY, _MISSING)
    enabled = getattr(opts, _KEY, False) if original is _MISSING else original
    if not enabled:
        yield False
        return

    # Forge serializes generation jobs. Do not call setters, callbacks, or
    # save(): this is a temporary override of the in-memory option only.
    data[_KEY] = False
    try:
        yield True
    finally:
        if original is _MISSING:
            data.pop(_KEY, None)
        else:
            data[_KEY] = original
