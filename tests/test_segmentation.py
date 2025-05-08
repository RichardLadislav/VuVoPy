# tests/test_segmented.py

import sys, os
# --- HACK: make pytest see your src/ folder; remove once your imports are fixed ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import pytest

from VuVoPy.data.containers.sample import VoiceSample
from VuVoPy.data.containers.prepocessing import Preprocessed
from VuVoPy.data.containers.segmentation import Segmented

def make_test_sample():
    """Create a simple VoiceSample → Preprocessed pipeline for segmentation tests."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    fs = 8000
    vs = VoiceSample(x, fs)
    pp = Preprocessed.from_voice_sample(vs)  # uses default α=0.94
    return vs, pp

def test_segmented_square_window_basic():
    vs, pp = make_test_sample()
    winlen, winover = 3, 1
    seg = Segmented.from_voice_sample(pp, winlen, 'square', winover)

    # ▶️ Check metadata
    assert seg.get_sampling_rate() == vs.get_sampling_rate()
    assert seg.get_window_length() == winlen
    assert seg.get_window_overlap() == winover
    assert seg.get_window_type() == 'square'

    # ▶️ Check output shape: (winlen, cols)
    expected_cols = int(np.ceil((pp.get_waveform().size - winover) / (winlen - winover)))
    assert seg.get_segment().shape == (winlen, expected_cols)
    assert seg.get_preem_segment().shape == (winlen, expected_cols)
    assert seg.get_norm_segment().shape == (winlen, expected_cols)

    # ▶️ Numeric check for “square” (no taper → just raw frames)
    # Frame indices: [[0,2], [1,3], [2,4]]
    np.testing.assert_allclose(
        seg.get_segment(),
        np.array([[0.0, 2.0],
                  [1.0, 3.0],
                  [2.0, 4.0]])
    )

    # Pre-emphasis was: [0, 1, 0.12, 1.12, 1.18], padded to length 6
    # so frames: [[0,0.12], [1,1.12], [0.12,1.18]]
    expected_pre = np.array([[0.0,  1.06],
                             [1.0,  1.12],
                             [1.06, 1.18]])
    np.testing.assert_allclose(seg.get_preem_segment(), expected_pre, atol=1e-6)

    # Normalization was x / 4 → [0, .25, .5, .75, 1.0], padded to 6
    expected_norm = np.array([[0.0, 0.5],
                              [0.25, 0.75],
                              [0.5, 1.0]])
    np.testing.assert_allclose(seg.get_norm_segment(), expected_norm)

def test_unknown_window_defaults_to_hamming():
    _, pp = make_test_sample()
    winlen, winover = 3, 1

    # Square segmentation (no taper)
    seg_sq = Segmented.from_voice_sample(pp, winlen, 'square', winover)
    # Unknown window name → should use hamming taper
    seg_hm = Segmented.from_voice_sample(pp, winlen, 'foobar', winover)

    # Hamming window of length 3
    win = np.hamming(winlen).reshape(-1, 1)

    # The “tapered” frames should equal the square‐window frames multiplied by the hamming window
    np.testing.assert_allclose(
        seg_hm.get_segment(),
        seg_sq.get_segment() * win
    )

def test_minimal_length_no_padding():
    # If signal length exactly equals winlen, we get one frame, no pad
    x = np.arange(5.0)  # length 5
    fs = 16000
    vs = VoiceSample(x, fs)
    pp = Preprocessed.from_voice_sample(vs)
    winlen, winover = 5, 2
    seg = Segmented.from_voice_sample(pp, winlen, 'square', winover)

    # Only one column since ceil((5 - 2)/(5 - 2)) == 1
    assert seg.get_segment().shape == (winlen, 1)
    np.testing.assert_allclose(seg.get_segment().flatten(), x * 1.0)  # square = raw

def test_invalid_window_type_metadata():
    # Even if window type is “invalid”, metadata should reflect it
    vs, pp = make_test_sample()
    seg = Segmented.from_voice_sample(pp, 4, 'noSuchWindow', 1)
    assert seg.get_window_type() == 'noSuchWindow'
    assert seg.get_window_length() == 4
    assert seg.get_window_overlap() == 1
