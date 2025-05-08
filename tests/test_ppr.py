import numpy as np
import pytest
from VuVoPy.features.utils.ppr import ppr
import VuVoPy.features.utils.ppr as pprmod
from VuVoPy.data.containers.sample import VoiceSample


class DummySegment:
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
def patch_vuvs(monkeypatch):
    # Save original Vuvs
    original = pprmod.vuvs

    # Fake Vuvs to inject total_silence_duration and capture threshold
    class FakeVuvs:
        def __init__(self, segment, fs, winlen, winover, wintype, smoothing_window):
            assert hasattr(segment, "get_sampling_rate")
            FakeVuvs._args = (fs, winlen, winover, wintype, smoothing_window)

        def get_total_silence_duration(self, min_silence_duration_ms):
            FakeVuvs._min_ms = min_silence_duration_ms
            return FakeVuvs.total_duration

    monkeypatch.setattr(pprmod, "vuvs", FakeVuvs)
    yield FakeVuvs
    monkeypatch.setattr(pprmod, "vuvs", original)


def test_ppr_basic(tmp_path, monkeypatch, patch_vuvs):
    # Setup dummy waveform and fs
    x = np.zeros(2000)
    fs = 1000

    # Stub vs.from_wav and pp.from_voice_sample
    monkeypatch.setattr(pprmod.vs, "from_wav", lambda path: VoiceSample(x, fs))
    monkeypatch.setattr(pprmod.pp, "from_voice_sample", lambda vsamp: vsamp)

    # Stub segmentation to use DummySegment
    monkeypatch.setattr(
        pprmod.sg,
        "from_voice_sample",
        lambda pp_obj, winlen, wintype, winover: DummySegment(fs, winlen, winover, wintype)
    )

    # Define total silence duration = 1s
    FakeVuvs = patch_vuvs
    FakeVuvs.total_duration = 1.0

    # Compute expected: silence fraction = 1.0 / (len(x)/fs) * 100 = 1/(2000/1000)*100 = 50%
    expected = 50.0

    result = ppr(str(tmp_path / "dummy.wav"), winlen=100, winover=50, wintype="hann", min_silence_duration_ms=100)
    # Verify FakeVuvs received threshold
    assert FakeVuvs._min_ms == 100
    assert pytest.approx(result, rel=1e-9) == expected


def test_ppr_no_silence(tmp_path, monkeypatch, patch_vuvs):
    x = np.ones(1500)
    fs = 1500

    monkeypatch.setattr(pprmod.vs, "from_wav", lambda path: VoiceSample(x, fs))
    monkeypatch.setattr(pprmod.pp, "from_voice_sample", lambda vsamp: vsamp)
    monkeypatch.setattr(
        pprmod.sg,
        "from_voice_sample",
        lambda pp_obj, winlen, wintype, winover: DummySegment(fs, winlen, winover, wintype)
    )

    FakeVuvs = patch_vuvs
    FakeVuvs.total_duration = 0.0

    # (len(x)/fs)=1s, silence=0 → percent=0
    result = ppr("ignored.wav")
    assert pytest.approx(result, abs=1e-12) == 0.0
