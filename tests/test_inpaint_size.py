from types import SimpleNamespace

import pytest

from aaaaaa.inpaint_size import align_inpaint_size, vae_spatial_alignment


def model(ratio):
    return SimpleNamespace(
        forge_objects=SimpleNamespace(vae=SimpleNamespace(downscale_ratio=ratio))
    )


@pytest.mark.parametrize(
    ("requested", "actual"),
    [
        ((896, 1192), (896, 1184)),
        ((1192, 896), (1184, 896)),
        ((896, 1192), (896, 1200)),
    ],
)
def test_generated_image_alignment(requested, actual):
    assert align_inpaint_size(requested, model(16), image_size=actual) == actual


def test_aligned_sdxl_is_unchanged():
    assert align_inpaint_size((896, 1192), model(8)) == (896, 1192)


def test_aligned_moss_is_unchanged():
    assert align_inpaint_size((896, 1200), model(16)) == (896, 1200)


def test_explicit_mask_scaled_minimum():
    assert align_inpaint_size(
        (513, 769), model(16), image_size=(512, 768), separate_size=True
    ) == (528, 784)


def test_hires_does_not_replace_base_size():
    assert align_inpaint_size((896, 1192), model(16), image_size=(1792, 2400)) == (
        896,
        1200,
    )


def test_video_vae_spatial_ratio():
    m = model((lambda a: a, 8, 16))
    assert vae_spatial_alignment(m) == (16, 8)
    assert align_inpaint_size((515, 517), m, separate_size=True) == (528, 520)


def test_unknown_model_is_unchanged():
    assert align_inpaint_size((896, 1192), SimpleNamespace()) == (896, 1192)
    assert align_inpaint_size((896, 1192), model(None)) == (896, 1192)


def test_checkpoint_override_uses_common_alignment():
    assert align_inpaint_size((896, 1192), model(8), model_override=True) == (896, 1200)


def test_small_positive_size():
    assert align_inpaint_size((1, 5), model(16)) == (16, 16)
