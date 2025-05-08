# tests/test_voicedsample.py
import numpy as np
import pytest

from VuVoPy.data.containers.sample import VoiceSample
from VuVoPy.data.containers.prepocessing import Preprocessed
from VuVoPy.data.containers.segmentation import Segmented
from VuVoPy.data.utils.vuvs_detection import Vuvs
from VuVoPy.data.containers.voiced_sample import VoicedSample

import sys
print(sys.path)
class DummyVuvs:
    """Simple stub that returns a fixed sequence of V/UV labels."""
    def __init__(self, labels):
        self._labels = labels

    def get_vuvs(self):
        return self._labels

def make_pipeline(x, fs, labels):
    """Builds a VoicedSample from raw x, fs and a sequence of vuv labels."""
    vs = VoiceSample(x, fs)
    pp = Preprocessed.from_voice_sample(vs)
    sg = Segmented.from_voice_sample(pp, winlen=3, wintype='square', winover=1)
    vuvs = DummyVuvs(labels)
    return VoicedSample(pp, vuvs, fs)

def test_label_stretch_preserves_length_and_proportions():
    # original labels length = 5, x length = 10
    labels = np.array([2,2,0,0,1])
    x = np.arange(10, dtype=float)
    fs = 1000
    vs = VoiceSample(x, fs)
    pp = Preprocessed.from_voice_sample(vs)
    vsamp = VoicedSample(pp, DummyVuvs(labels), fs)

    stretched = vsamp.label_stretch()
    # must match target length
    assert stretched.shape[0] == x.shape[0]

    # counts proportional: original segment lens [2,2,1] → proportions [2/5,2/5,1/5]
    # expect counts [4,4,2]
    unique, counts = np.unique(stretched, return_counts=True)
    count_map = dict(zip(unique, counts))
    assert count_map[2] == 4
    assert count_map[0] == 4
    assert count_map[1] == 2

def test_get_voiced_sample_picks_label_2():
    # use the same labels as above
    labels = np.array([2,2,0,0,1])
    x = np.linspace(0, 9, 10)
    fs = 1000
    vsamp = make_pipeline(x, fs, labels)

    voiced = vsamp.get_waveform()  # alias for voiced_sample
    # since label_stretch yields 4 of label 2, we expect first 4 samples of x
    assert np.allclose(voiced, x[:4])

def test_get_silence_remove_sample_removes_long_silence():
    # Here fs small so min_frames = ceil(50/1000 * 2) = 1, so any zero runs removed
    labels = np.array([0,0,2,2,0])
    x = np.array([10, 20, 30, 40, 50], dtype=float)
    fs = 2
    vsamp = make_pipeline(x, fs, labels)

    sr = vsamp.get_silence_remove_sample()
    # segments where label==0 get removed entirely, so only the two 2s remain
    assert np.allclose(sr, x[2:4])

def test_get_silence_remove_sample_keeps_short_silence():
    # Here fs large so min_frames = ceil(50/1000*1000) = 50, zero runs of length <50 kept
    labels = np.array([0,0,2,2,0])
    x = np.arange(5, dtype=float)
    fs = 1000
    vsamp = make_pipeline(x, fs, labels)

    sr = vsamp.get_silence_remove_sample()
    # since no zero-run is ≥50 frames, nothing is removed → same as voiced_sample
    # voiced_sample for this label run = 2s only → x positions 2 and 3
    assert np.allclose(vsamp.get_voiced_sample(), x[2:4])
    # but silence_remove should leave all x unchanged because zero-runs too short
    # the code currently removes only runs >= min_frames, so here min_frames=50, no removal
    assert np.allclose(sr, x)

def test_get_sampling_rate():
    x = np.zeros(10)
    fs = 12345
    vsamp = make_pipeline(x, fs, np.zeros(5, dtype=int))
    assert vsamp.get_sampling_rate() == fs
