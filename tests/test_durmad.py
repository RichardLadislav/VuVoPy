import numpy as np
import pytest
from VuVoPy.features.utils.durmad import durmad
import VuVoPy.features.utils.durmad as durmod
from VuVoPy.data.containers.sample import VoiceSample


@pytest.fixture(autouse=True)
def patch_vuvs_detection(monkeypatch):
    # 1) Save the real Vuvs class
    original_vuvs = durmod.vuvs

    # 2) Define a FakeVuvs for testing
    class FakeVuvs:
        def __init__(self, segment, fs, winlen, winover, wintype, smoothing_window):
            # ensure the real pipeline ran
            assert hasattr(segment, "get_segment")
            # we will set FakeVuvs.durations in each test
        def get_silence_durations(self):
            return FakeVuvs.durations

    # 3) Monkey‐patch it in
    monkeypatch.setattr(durmod, "vuvs", FakeVuvs)
    yield
    # 4) Restore the original class after each test
    monkeypatch.setattr(durmod, "vuvs", original_vuvs)


def test_durmad_with_known_durations(tmp_path, monkeypatch):
    # stub out VoiceSample.from_wav to use a trivial waveform
    x = np.zeros(1000); fs = 1000
    monkeypatch.setattr(
        durmod.vs,
        "from_wav",
        lambda path: VoiceSample(x, fs)
    )

    # inject our test durations
    FakeVuvs = durmod.vuvs
    FakeVuvs.durations = np.array([1.0, 3.0, 5.0])

    result = durmad(str(tmp_path / "dummy.wav"),
                    winlen=100, winover=50, wintype="hann")

    # verify mean absolute deviation from median 3.0 → [2,0,2] → 4/3
    assert pytest.approx(result, rel=1e-9) == (2 + 0 + 2) / 3

