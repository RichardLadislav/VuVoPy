import numpy as np
import pytest
from unittest import mock
import librosa
#import sys
#import os
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from VuVoPy.data.containers.sample import VoiceSample as vs

def test_initialization():
    x = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    fs = 16000
    sample = vs(x, fs)
    assert isinstance(sample, vs)
    assert np.array_equal(sample.get_waveform(), x)
    assert sample.get_sampling_rate() == fs

@mock.patch("librosa.load")
def test_from_wav(mock_load):
    dummy_waveform = np.array([0.1, 0.2, 0.3])
    dummy_fs = 22050
    mock_load.return_value = (dummy_waveform, dummy_fs)

    sample = vs.from_wav("dummy.wav")
    assert isinstance(sample, vs)
    assert np.array_equal(sample.get_waveform(), dummy_waveform)
    assert sample.get_sampling_rate() == dummy_fs

def test_get_waveform_and_sampling_rate():
    x = np.linspace(-1, 1, 1000)
    fs = 44100
    sample = vs(x, fs)
    assert isinstance(sample.get_waveform(), np.ndarray)
    assert sample.get_sampling_rate() == 44100
