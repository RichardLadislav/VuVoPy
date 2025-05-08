import numpy as np
import pytest
from VuVoPy.features.utils.hnr import hnr
import VuVoPy.features.utils.hnr as hnrmod

class DummySegment:
    def __init__(self, norm_array, fs):
        # norm_array shape = (frame_length, num_frames)
        self._norm = norm_array
        self._fs = fs
    def get_norm_segment(self):
        return self._norm
    def get_sampling_rate(self):
        return self._fs


@pytest.fixture(autouse=True)
def patch_pipeline(monkeypatch):
    # 1) Stub vs.from_wav → no-op
    monkeypatch.setattr(hnrmod.vs, "from_wav", lambda path: None)
    # 2) Stub pp.from_voice_sample → no-op
    monkeypatch.setattr(hnrmod.pp, "from_voice_sample", lambda vs_obj: None)
    # 3) Stub sg.from_voice_sample → returns our DummySegment, but we’ll override per-test
    def fake_from_voice_sample(pp_obj, winlen, wintype, winover):
        # Will be replaced in individual tests
        raise RuntimeError("Use per-test monkeypatch")
    monkeypatch.setattr(hnrmod.sg, "from_voice_sample", fake_from_voice_sample)
    yield
    # no teardown needed

def test_hnr_all_silent(tmp_path, monkeypatch):
    """If all frames are silent (max abs < 1e-6), hnr() should return nan."""
    fs = 1000
    # Create a silent segment: frame_length=10, num_frames=3
    norm = np.zeros((10, 3))
    monkeypatch.setattr(
        hnrmod.sg,
        "from_voice_sample",
        lambda pp_obj, winlen, wintype, winover: DummySegment(norm, fs)
    )

    result = hnr("ignored.wav", winlen=10, winover=5, wintype="hann")
    assert np.isnan(result)

def test_hnr_impulse_frame(tmp_path, monkeypatch):
    """
    One silent frame, one impulse frame → only impulse yields HNR.
    Expect HNR = 10*log10(r_max/(1-r_max)) with r_max clipped to [1e-4,0.999].
    For an impulse, autocorr peak at lag>0 is 0 → clipped to 1e-4.
    So HNR = 10*log10(1e-4/(1-1e-4)).
    """
    fs = 500  # arbitrary
    # Create norm array: 2 frames of length 8
    norm = np.zeros((8, 2))
    # First frame all zeros (silent), second frame impulse at index 0
    norm[0, 1] = 1.0

    monkeypatch.setattr(
        hnrmod.sg,
        "from_voice_sample",
        lambda pp_obj, winlen, wintype, winover: DummySegment(norm, fs)
    )

    # Compute expected HNR
    r_max = 1e-4
    expected_hnr = 10 * np.log10(r_max / (1 - r_max))

    result = hnr("ignored.wav", winlen=8, winover=7, wintype="hann", f0_min=50, f0_max=200)
    # Only one non-silent frame → mean = that frame’s HNR
    assert pytest.approx(result, rel=1e-6) == expected_hnr
