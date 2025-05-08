# tests/test_preprocessed.py

import sys, os
# --- HACK: make sure pytest can see your src/ folder; remove once imports are fixed ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import pytest
from VuVoPy.data.containers.sample import VoiceSample
from VuVoPy.data.containers.prepocessing import Preprocessed


def make_simple_sample():
    # waveform [0, 1, -1], fs = 1000
    x = np.array([0.0, 1.0, -1.0], dtype=float)
    fs = 1000
    return VoiceSample(x, fs)


def test_from_voice_sample_default_alpha():
    vs = make_simple_sample()
    pp = Preprocessed.from_voice_sample(vs)  # default alpha = 0.94

    # original should be unchanged
    assert np.allclose(pp.get_waveform(), vs.get_waveform())

    # normalization: divide by max(abs(x)) == 1.0 → same as original
    expected_norm = vs.get_waveform() / 1.0
    assert np.allclose(pp.get_normalization(), expected_norm)

    # pre-emphasis: [x0,
    #               x1 - α*x0 = 1 - 0.94*0 = 1,
    #               x2 - α*x1 = -1 - 0.94*1 = -1.94]
    α = pp.alpha
    expected_preem = np.array([
        vs.x[0],
        vs.x[1] - α * vs.x[0],
        vs.x[2] - α * vs.x[1],
    ])
    assert np.allclose(pp.get_preemphasis(), expected_preem)


def test_from_voice_sample_custom_alpha():
    vs = make_simple_sample()
    custom_alpha = 0.5
    pp = Preprocessed.from_voice_sample(vs, alpha=custom_alpha)

    # verify the stored alpha
    assert pp.alpha == custom_alpha

    # check pre-emphasis with custom alpha
    expected_preem = np.array([
        vs.x[0],
        vs.x[1] - custom_alpha * vs.x[0],
        vs.x[2] - custom_alpha * vs.x[1],
    ])
    # default get_preemphasis() uses stored alpha
    assert np.allclose(pp.get_preemphasis(), expected_preem)

    # and get_preemphasis(alpha=…) overrides correctly
    override_alpha = 0.2
    expected_override = np.array([
        vs.x[0],
        vs.x[1] - override_alpha * vs.x[0],
        vs.x[2] - override_alpha * vs.x[1],
    ])
    assert np.allclose(pp.get_preemphasis(alpha=override_alpha), expected_override)


def test_get_waveform_and_sampling_rate():
    vs = make_simple_sample()
    pp = Preprocessed.from_voice_sample(vs)

    assert np.shares_memory(pp.get_waveform(), vs.get_waveform()) or np.allclose(pp.get_waveform(), vs.get_waveform())
    assert pp.get_sampling_rate() == vs.get_sampling_rate()


def test_zero_signal_behavior():
    # edge-case: all zeros waveform
    x = np.zeros(5)
    fs = 8000
    vs = VoiceSample(x, fs)
    pp = Preprocessed.from_voice_sample(vs, alpha=0.7)

    # normalization: max(abs(x)) == 0 → xnorm should be original zeros
    assert np.allclose(pp.get_normalization(), x)

    # pre-emphasis on zeros yields zeros
    assert np.allclose(pp.get_preemphasis(), np.zeros_like(x))


def test_manual_init_defaults():
    # test __init__ fallback when xnorm or preem are None
    x = np.array([1.0, -2.0, 3.0])
    fs = 22050
    # both xnorm and preem None → should default to x
    pp = Preprocessed(x, fs, xnorm=None, preem=None, alpha=0.33)

    assert np.allclose(pp.get_waveform(), x)
    assert np.allclose(pp.get_normalization(), x)
    assert np.allclose(pp.get_preemphasis(), x)
    assert pp.alpha == 0.33
