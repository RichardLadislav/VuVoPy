# tests/test_durmed.py

import sys, os
import numpy as np
import pytest

# until VuVoPy is installed in editable mode:
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from VuVoPy.features.utils.durmed import durmed
import VuVoPy.features.utils.durmed as durmod
from VuVoPy.data.containers.sample import VoiceSample


@pytest.fixture(autouse=True)
def patch_vuvs_detection(monkeypatch):
    # 1) Save the real Vuvs
    original_vuvs = durmod.vuvs

    # 2) FakeVuvs lets us inject .durations
    class FakeVuvs:
        def __init__(self, segment, fs, winlen, winover, wintype, smoothing_window):
            # ensure segmentation actually ran
            assert hasattr(segment, "get_segment")
        def get_silence_durations(self):
            return FakeVuvs.durations

    # 3) Patch into the module under test
    monkeypatch.setattr(durmod, "vuvs", FakeVuvs)
    yield
    # 4) Restore original
    monkeypatch.setattr(durmod, "vuvs", original_vuvs)


def test_durmed_with_known_durations(tmp_path, monkeypatch):
    # stub VoiceSample.from_wav → trivial waveform
    x = np.arange(500, dtype=float)
    fs = 1000
    monkeypatch.setattr(
        durmod.vs,
        "from_wav",
        lambda path: VoiceSample(x, fs)
    )

    # define our test durations
    FakeVuvs = durmod.vuvs
    FakeVuvs.durations = np.array([2.0, 4.0, 6.0, 8.0])

    # run
    result = durmed(str(tmp_path / "dummy.wav"),
                    winlen=100, winover=50, wintype="hann")

    # expected median = (4+6)/2 = 5; so durmed returns 5.0
    assert pytest.approx(result, rel=1e-9) == 5.0

