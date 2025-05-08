import numpy as np
import pytest
from VuVoPy.features.utils.duv import duv
import VuVoPy.features.utils.duv as duvmod
from VuVoPy.data.containers.sample import VoiceSample


@pytest.fixture(autouse=True)
def patch_vuvs(monkeypatch):
    # Save the original
    original = duvmod.vuvs

    # FakeVuvs: we only care about .get_vuvs()
    class FakeVuvs:
        def __init__(self, segment, fs, winlen, winover, wintype, smoothing_window):
            # pipeline must have run
            assert hasattr(segment, "get_segment")
        def get_vuvs(self):
            return FakeVuvs.labels

    # Patch in
    monkeypatch.setattr(duvmod, "vuvs", FakeVuvs)
    yield FakeVuvs
    # Restore
    monkeypatch.setattr(duvmod, "vuvs", original)


def test_duv_known_labels(tmp_path, monkeypatch, patch_vuvs):
    # Stub out VoiceSample.from_wav to return a trivial sample
    x = np.zeros(100)
    fs = 1000
    monkeypatch.setattr(
        duvmod.vs,
        "from_wav",
        lambda path: VoiceSample(x, fs)
    )

    # Define labels: 1=unvoiced, 2=voiced, 0=silence
    labels = np.array([1,2,1,0,1,2,1])
    # so count of label==1 is 4 out of 7 → ~57.142857%
    FakeVuvs = patch_vuvs
    FakeVuvs.labels = labels

    # Call duv
    result = duv(str(tmp_path / "dummy.wav"),
                 winlen=50, winover=25, wintype="hann")

    expected = (np.sum(labels == 1) / len(labels)) * 100
    assert pytest.approx(result, rel=1e-9) == expected



def test_duv_all_ones(tmp_path, monkeypatch, patch_vuvs):
    x = np.arange(20, dtype=float)
    fs = 8000
    monkeypatch.setattr(
        duvmod.vs,
        "from_wav",
        lambda path: VoiceSample(x, fs)
    )

    # labels all 1s
    labels = np.ones(10, dtype=int)
    FakeVuvs = patch_vuvs
    FakeVuvs.labels = labels

    result = duv("ignored2.wav", winlen=10, winover=5, wintype="square")
    assert pytest.approx(result, abs=1e-12) == 100.0

