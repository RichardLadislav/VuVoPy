import numpy as np
import pytest
from VuVoPy.features.utils.shimmer import shimmerAPQ
import VuVoPy.features.utils.shimmer as mod
from VuVoPy.data.containers.sample import VoiceSample

# Fake segment factory
def FakeSegment(fs, winlen, winover, wintype):
    class S:
        def get_sampling_rate(self): return fs
        def get_window_length(self): return winlen
        def get_window_overlap(self): return winover
        def get_window_type(self): return wintype
    return S()

@pytest.fixture(autouse=True)
def patch_deps(monkeypatch):
    # Save originals
    orig_vs = mod.vs.from_wav
    orig_pp = mod.pp.from_voice_sample
    orig_sg = mod.sg.from_voice_sample
    orig_vuvs = mod.vuvs
    orig_vos = mod.vos
    orig_f0 = mod.f0

    # Patch vs.from_wav and pp.from_voice_sample
    monkeypatch.setattr(mod.vs, "from_wav", lambda path: None)
    monkeypatch.setattr(mod.pp, "from_voice_sample", lambda x: None)

    # Patch segmentation
    monkeypatch.setattr(
        mod.sg, "from_voice_sample",
        lambda pp_obj, winlen, wintype, winover: FakeSegment(1000, winlen, winover, wintype)
    )

    # Patch vuvs (unused by shimmerAPQ but must exist)
    monkeypatch.setattr(mod, "vuvs", lambda *args, **kwargs: None)

    # Patch VoicedSample (vos)
    class FakeVos:
        def __init__(self, pre, labels, fs): pass
        def get_silence_remove_sample(self): return FakeVos._signal
        def get_sampling_rate(self): return FakeVos._fs
    monkeypatch.setattr(mod, "vos", FakeVos)

    # Patch FundamentalFrequency (f0)
    class FakeF0:
        def __init__(self, sample, plim, sTHR): pass
        def get_f0(self): return FakeF0._f0_array
    monkeypatch.setattr(mod, "f0", FakeF0)

    yield FakeVos, FakeF0

    # Restore originals
    monkeypatch.setattr(mod.vs, "from_wav", orig_vs)
    monkeypatch.setattr(mod.pp, "from_voice_sample", orig_pp)
    monkeypatch.setattr(mod.sg, "from_voice_sample", orig_sg)
    monkeypatch.setattr(mod, "vuvs", orig_vuvs)
    monkeypatch.setattr(mod, "vos", orig_vos)
    monkeypatch.setattr(mod, "f0", orig_f0)


def test_shimmerapq_few_f0(patch_deps):
    FakeVos, FakeF0 = patch_deps
    # Fewer F0 points than n_points => return 0
    FakeF0._f0_array = np.array([100.0])
    FakeVos._signal = np.arange(100)
    FakeVos._fs = 1000
    result = shimmerAPQ("ignored.wav", n_points=3)
    assert result == 0


def test_shimmerapq_constant_signal(patch_deps):
    FakeVos, FakeF0 = patch_deps
    # Enough F0 points
    FakeF0._f0_array = np.ones(5) * 200.0
    # Constant signal => amplitude differences zero => return 0
    FakeVos._signal = np.ones(500)
    FakeVos._fs = 1000
    result = shimmerAPQ("ignored.wav", n_points=3)
    assert result == 0
