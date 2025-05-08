import numpy as np
import pytest
from VuVoPy.features.utils.relseosd import relSEOSD
import VuVoPy.features.utils.relseosd as mod
from VuVoPy.data.containers.sample import VoiceSample

# Define a fake segment with required methods
class FakeSegment:
    def __init__(self, fs, winlen, winover, wintype):
        self._fs = fs
        self._winlen = winlen
        self._winover = winover
        self._wintype = wintype

    def get_sampling_rate(self):
        return self._fs

    def get_window_length(self):
        return self._winlen

    def get_window_overlap(self):
        return self._winover

    def get_window_type(self):
        return self._wintype

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Save originals
    orig_vs = mod.vs.from_wav
    orig_pp = mod.pp.from_voice_sample
    orig_sg = mod.sg.from_voice_sample
    orig_vuvs = mod.vuvs
    orig_vos = mod.vos
    orig_rms = mod.lib.feature.rms

    # Patch vs.from_wav and pp.from_voice_sample
    monkeypatch.setattr(mod.vs, "from_wav", lambda path: None)
    monkeypatch.setattr(mod.pp, "from_voice_sample", lambda v: None)

    # Patch segmentation to return FakeSegment with known fs, passing through winlen, wintype, winover
    fake_fs = 1000
    monkeypatch.setattr(
        mod.sg,
        "from_voice_sample",
        lambda pp_obj, winlen, wintype, winover: FakeSegment(fake_fs, winlen, winover, wintype)
    )

    # Patch vuvs to dummy
    class FakeVuvs:
        def __init__(self, segment, fs, winlen, winover, wintype, smoothing_window):
            pass
    monkeypatch.setattr(mod, "vuvs", FakeVuvs)

    # Patch voiced sample removal
    class FakeVos:
        def __init__(self, preprocessed, labels, fs):
            pass
        def get_silence_remove_sample(self):
            return FakeVos.sample
    monkeypatch.setattr(mod, "vos", FakeVos)

    # Patch RMS feature
    monkeypatch.setattr(
        mod.lib.feature, "rms",
        lambda y, frame_length, hop_length, center: FakeVos.rms_contour
    )

    yield { "fake_fs": fake_fs, "FakeVos": FakeVos }

    # Restore originals
    monkeypatch.setattr(mod.vs, "from_wav", orig_vs)
    monkeypatch.setattr(mod.pp, "from_voice_sample", orig_pp)
    monkeypatch.setattr(mod.sg, "from_voice_sample", orig_sg)
    monkeypatch.setattr(mod, "vuvs", orig_vuvs)
    monkeypatch.setattr(mod, "vos", orig_vos)
    monkeypatch.setattr(mod.lib.feature, "rms", orig_rms)


def test_relSEOSD_zero_mean(patch_dependencies):
    # silence_removed_sample and rms contour of zeros => mean=0 => result=0
    FakeVos = patch_dependencies["FakeVos"]
    FakeVos.sample = np.zeros(4)
    FakeVos.rms_contour = np.zeros(4)

    result = relSEOSD("ignored.wav", winlen=400, winover=200)
    assert result == 0.0


def test_relSEOSD_constant(patch_dependencies):
    # contour constant non-zero: std=0 => result=0
    FakeVos = patch_dependencies["FakeVos"]
    FakeVos.sample = np.ones(5)
    FakeVos.rms_contour = np.ones(5)

    result = relSEOSD("ignored.wav")
    assert pytest.approx(result, abs=1e-12) == 0.0


def test_relSEOSD_nontrivial(patch_dependencies):
    # contour with variability
    FakeVos = patch_dependencies["FakeVos"]
    FakeVos.sample = np.array([0.1, 0.2, 0.3, 0.4])
    FakeVos.rms_contour = FakeVos.sample

    mean_val = np.mean(FakeVos.rms_contour)
    std_val = np.std(FakeVos.rms_contour)
    expected = std_val / mean_val

    result = relSEOSD("ignored.wav", winlen=512, winover=256)
    assert pytest.approx(result, rel=1e-9) == expected
