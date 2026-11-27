from types import SimpleNamespace

import pytest

from aaaaaa.anima_reference import is_anima_engine, suspend_anima_reference

Anima = type("Anima", (), {"__module__": "backend.diffusion_engine.anima"})
Krea = type("Krea", (), {"__module__": "backend.diffusion_engine.krea"})
KEY = "anima_do_reference"


class Options:
    def __init__(self, data, default=False):
        self.data = data
        self.default = default

    def __getattr__(self, key):
        if key == KEY:
            return self.data.get(key, self.default)
        raise AttributeError(key)


def test_engine_detection():
    assert is_anima_engine(Anima())
    assert is_anima_engine(type("Subclass", (Anima,), {})())
    assert not is_anima_engine(Krea())
    assert not is_anima_engine(None)
    assert not is_anima_engine(type("Anima", (), {})())


def test_enabled_anima_reference_restored():
    opts = Options({KEY: True, "krea2_do_reference": True})
    before = opts.data.copy()
    with suspend_anima_reference(opts, Anima()) as changed:
        assert changed
        assert not opts.anima_do_reference
        assert opts.data["krea2_do_reference"] is True
    assert opts.data == before


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_exception_restores_setting(error):
    opts = Options({KEY: True})

    def fail_during_inpaint():
        with suspend_anima_reference(opts, Anima()):
            assert opts.data[KEY] is False
            msg = "test"
            raise error(msg)

    with pytest.raises(error, match="test"):
        fail_during_inpaint()
    assert opts.data[KEY] is True


def test_already_disabled_is_unchanged():
    opts = Options({KEY: False})
    with suspend_anima_reference(opts, Anima()) as changed:
        assert not changed
        assert opts.data == {KEY: False}
    assert opts.data == {KEY: False}


def test_other_model_is_unchanged():
    opts = Options({KEY: True, "krea2_do_reference": True})
    with suspend_anima_reference(opts, Krea()) as changed:
        assert not changed
        assert opts.data[KEY] is True
        assert opts.data["krea2_do_reference"] is True


def test_missing_key_default_restored_without_creating_setting():
    opts = Options({}, default=True)
    with suspend_anima_reference(opts, Anima()) as changed:
        assert changed
        assert opts.anima_do_reference is False
    assert opts.data == {}
    assert opts.anima_do_reference is True


def test_missing_option_and_non_forge_noop():
    opts = SimpleNamespace(data={})
    with suspend_anima_reference(opts, Anima()) as changed:
        assert not changed
        assert opts.data == {}
    with suspend_anima_reference(SimpleNamespace(), Anima()) as changed:
        assert not changed


def test_checkpoint_override_guards_later_loaded_anima():
    opts = Options({KEY: True, "krea2_do_reference": True})
    with suspend_anima_reference(opts, Krea(), checkpoint_override=True) as changed:
        assert changed
        assert not opts.anima_do_reference
        assert opts.data["krea2_do_reference"] is True
    assert opts.anima_do_reference


def test_nested_scope_preserves_outer_override():
    opts = Options({KEY: True})
    with suspend_anima_reference(opts, Anima()) as outer:
        assert outer
        with suspend_anima_reference(opts, Anima()) as inner:
            assert not inner
        assert not opts.anima_do_reference
    assert opts.anima_do_reference
