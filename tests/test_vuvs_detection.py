# tests/test_vuvs_detection.py
import numpy as np
import pytest
from types import SimpleNamespace
import VuVoPy.data.utils.vuvs_detection as det_mod
from VuVoPy.data.utils.vuvs_detection import Vuvs

# Stubbed vuvs_gmm module
class DummySeg:
    def __init__(self, labels, fs):
        self._labels = labels
        self._fs = fs
    def get_segment(self): return None
    def get_preem_segment(self): return None
    def get_norm_segment(self): return None
    def get_sampling_rate(self): return self._fs

def test_vuvs_detection_basic(monkeypatch):
    # Monkeypatch vuvs_gmm to return a known label sequence
    expected = np.array([0,2,2,1,0])
    def fake_vuvs_gmm(seg, fs, winover, smoothing):
        return expected
    monkeypatch.setattr(det_mod, "vuvs_gmm", fake_vuvs_gmm)

    fs = 1000
    dummy_seg = DummySeg(labels=expected, fs=fs)
    v = Vuvs(dummy_seg, fs=fs, winlen=100, winover=50, smoothing_window=7)

    # get_vuvs should return exactly what fake_vuvs_gmm produced
    np.testing.assert_array_equal(v.get_vuvs(), expected)
    assert v.get_sampling_rate() == fs

def test_silence_statistics(monkeypatch):
    # Label pattern: zeros of lengths [2,1], hop_duration=(winlen-winover)/fs
    labels = np.array([0,0,2,2,0,0,0,1])
    fs = 2
    winlen, winover = 4, 2
    hop_duration = (winlen - winover) / fs  # = (4-2)/2 = 1s per frame

    # Silence segments: [0,0] length2→2s, and [0,0,0] length3→3s
    expected_durations_default = [2.0, 3.0]
    expected_total_default = 5.0
    expected_count_default = 2

    # Monkeypatch vuvs_gmm
    monkeypatch.setattr(det_mod, "vuvs_gmm", lambda *args: labels)

    v = Vuvs(DummySeg(None, fs), fs=fs, winlen=winlen, winover=winover)

    # Default threshold=50ms→min_frames=ceil(0.05/1)=1
    count = v.get_silence_count()
    durations = v.get_silence_durations()
    total = v.get_total_silence_duration()

    assert count == expected_count_default
    np.testing.assert_allclose(durations, expected_durations_default)
    assert pytest.approx(total) == expected_total_default

    # Custom threshold 2500ms→min_frames=ceil(2.5/1)=3
    expected_durations_custom = [3.0]
    expected_total_custom = 3.0
    expected_count_custom = 1

    count2 = v.get_silence_count(min_silence_duration_ms=2500)
    durations2 = v.get_silence_durations(min_silence_duration_ms=2500)
    total2 = v.get_total_silence_duration(min_silence_duration_ms=2500)

    assert count2 == expected_count_custom
    np.testing.assert_allclose(durations2, expected_durations_custom)
    assert pytest.approx(total2) == expected_total_custom
