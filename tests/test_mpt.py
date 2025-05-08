import os
import numpy as np
import pytest
from VuVoPy.features.utils.mpt import mpt
import VuVoPy.features.utils.mpt as mptmod
from VuVoPy.data.containers.sample import VoiceSample


@pytest.fixture(autouse=True)
def patch_vuvs(monkeypatch):
    # 1) Save original Vuvs
    original_vuvs = mptmod.vuvs

    # 2) FakeVuvs: allow injection of label sequences
    class FakeVuvs:
        def __init__(self, segment, fs, winlen, winover, wintype, smoothing_window):
            # ensure segmentation ran
            assert hasattr(segment, "get_segment")
        def get_vuvs(self):
            return FakeVuvs.labels

    # 3) Patch into module under test
    monkeypatch.setattr(mptmod, "vuvs", FakeVuvs)
    yield FakeVuvs

    # 4) Restore original
    monkeypatch.setattr(mptmod, "vuvs", original_vuvs)


def test_mpt_basic(monkeypatch, patch_vuvs):
    # Stub vs.from_wav to return a dummy VoiceSample
    x = np.zeros(1000)
    fs = 1000
    monkeypatch.setattr(
        mptmod.vs,
        "from_wav",
        lambda path: VoiceSample(x, fs)
    )

    # Define labels: count frames == 2 (voiced)
    FakeVuvs = patch_vuvs
    FakeVuvs.labels = np.array([2, 0, 2, 2, 1])  # three voiced frames

    winlen = 100
    winover = 50
    expected = (np.sum(FakeVuvs.labels == 2) * (winlen - winover)) / fs

    result = mpt("ignored.wav", winlen=winlen, winover=winover, wintype="hann")
    assert pytest.approx(result, rel=1e-9) == expected


def test_mpt_no_voiced(monkeypatch, patch_vuvs):
    # Stub vs.from_wav again
    x = np.ones(200)
    fs = 8000
    monkeypatch.setattr(
        mptmod.vs,
        "from_wav",
        lambda path: VoiceSample(x, fs)
    )

    FakeVuvs = patch_vuvs
    FakeVuvs.labels = np.array([0, 1, 1, 0, 1])  # no voiced frames

    result = mpt("ignored.wav", winlen=100, winover=50, wintype="hann")
    assert pytest.approx(result, abs=1e-12) == 0.0
