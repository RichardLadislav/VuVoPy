import numpy as np
import pytest
from types import SimpleNamespace
from VuVoPy.features.utils.spir import spir
import VuVoPy.features.utils.spir as spirmod

# Fake segment class with required methods
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
    orig_vs = spirmod.vs.from_wav
    orig_pp = spirmod.pp.from_voice_sample
    orig_sg = spirmod.sg.from_voice_sample
    orig_vuvs = spirmod.vuvs

    # Stub vs.from_wav → no-op
    monkeypatch.setattr(spirmod.vs, "from_wav", lambda path: None)

    # Stub pp.from_voice_sample → returns object with get_waveform
    waveform = np.arange(1000)
    fake_pre = SimpleNamespace(get_waveform=lambda: waveform)
    monkeypatch.setattr(spirmod.pp, "from_voice_sample", lambda vsamp: fake_pre)

    # Stub segmentation to return FakeSegment with known fs
    fake_fs = 100
    monkeypatch.setattr(
        spirmod.sg,
        "from_voice_sample",
        lambda pp_obj, winlen, wintype, winover: FakeSegment(fake_fs, winlen, winover, wintype)
    )

    # Stub vuvs to return controlled silence count
    class FakeVuvs:
        def __init__(self, segment, fs, winlen, winover, wintype, smoothing_window):
            pass
        def get_silence_count(self):
            return FakeVuvs.count

    monkeypatch.setattr(spirmod, "vuvs", FakeVuvs)

    yield FakeVuvs, waveform, fake_fs

    # Restore originals
    monkeypatch.setattr(spirmod.vs, "from_wav", orig_vs)
    monkeypatch.setattr(spirmod.pp, "from_voice_sample", orig_pp)
    monkeypatch.setattr(spirmod.sg, "from_voice_sample", orig_sg)
    monkeypatch.setattr(spirmod, "vuvs", orig_vuvs)


def test_spir_basic(patch_dependencies):
    FakeVuvs, waveform, fs = patch_dependencies
    FakeVuvs.count = 25  # 25 silent frames
    # Silence percentage = 25 / (len(waveform)/fs)
    expected = 25 / (len(waveform) / fs)
    result = spir("ignored.wav", winlen=200, winover=100)
    assert pytest.approx(result, rel=1e-9) == expected


def test_spir_no_silence(patch_dependencies):
    FakeVuvs, waveform, fs = patch_dependencies
    FakeVuvs.count = 0
    result = spir("ignored.wav")
    assert pytest.approx(result, abs=1e-12) == 0.0
