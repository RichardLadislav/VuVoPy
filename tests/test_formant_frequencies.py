# tests/test_formant_frequencies.py

import numpy as np
import pytest
from VuVoPy.data.utils.formant_frequencies import FormantFrequencies

class DummySegments:
    """A stub that mimics the minimal interface used by FormantFrequencies."""
    def __init__(self, num_segments, seg_length, fs):
        # segment arrays of shape (seg_length, num_segments)
        self._seg = np.zeros((seg_length, num_segments))
        self._fs = fs

    def get_segment(self):
        return self._seg

    def get_preem_segment(self):
        return self._seg

    def get_norm_segment(self):
        return self._seg

    def get_sampling_rate(self):
        return self._fs

def test_zero_formants_path(monkeypatch):
    # Arrange: a DummySegments with 2 segments of length 5, fs=1000
    num_segments = 2
    seg_length = 5
    fs = 1000
    dummy = DummySegments(num_segments, seg_length, fs)

    # Monkeypatch librosa.lpc to return a 2×3 coefficient array
    # so roots are real (no imag>0), triggering the empty‐roots branch
    def fake_lpc(data, order):
        # ignore 'data' and 'order'; return two identical quadratic polys
        return np.tile(np.array([1.0, -2.0, 1.0]), (num_segments, 1))

    monkeypatch.setattr(
        "VuVoPy.data.utils.formant_frequencies.lb.lpc",
        fake_lpc
    )

    # Act
    ff_obj = FormantFrequencies.from_voice_sample(dummy)

    # Assert sampling rate
    assert ff_obj.get_sampling_rate() == fs

    # All formant frequencies should be zeros, shape (num_segments, 3)
    f_raw = ff_obj.get_formants()
    f_pre = ff_obj.get_formants_preem()
    f_norm = ff_obj.get_formants_norm()

    assert f_raw.shape == (num_segments, 3)
    assert f_pre.shape == (num_segments, 3)
    assert f_norm.shape == (num_segments, 3)

    np.testing.assert_array_equal(f_raw, np.zeros((num_segments, 3)))
    np.testing.assert_array_equal(f_pre, np.zeros((num_segments, 3)))
    np.testing.assert_array_equal(f_norm, np.zeros((num_segments, 3)))

def test_order_and_empty_data(monkeypatch):
    # If get_segment() returns an empty array, the code should still handle it
    class EmptySegments(DummySegments):
        def __init__(self):
            self._seg = np.zeros((0, 0))
            self._fs = 8000

    empty = EmptySegments()

    # Monkeypatch lpc so it’s not called on empty data
    called = {"count": 0}
    def fake_lpc(data, order):
        called["count"] += 1
        return np.zeros((0, 3))  # zero‐row, three‐col
    monkeypatch.setattr(
        "VuVoPy.data.utils.formant_frequencies.lb.lpc",
        fake_lpc
    )

    ff_obj = FormantFrequencies.from_voice_sample(empty)

    # lpc should be called three times (raw, preem, norm)
    assert called["count"] == 3

    # Empty formant arrays
    assert ff_obj.get_formants().size == 0
    assert ff_obj.get_formants_preem().size == 0
    assert ff_obj.get_formants_norm().size == 0

    # Sampling rate still returned correctly
    assert ff_obj.get_sampling_rate() == 8000
