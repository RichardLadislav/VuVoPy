# tests/test_fundamental_frequency.py

import numpy as np
import pytest

# Import your class under test
from VuVoPy.data.utils.fundamental_frequency import FundamentalFrequency


class DummySample:
    """A minimal stand-in for VoiceSample (or any subclass)."""
    def __init__(self, x, fs):
        self._x = x
        self._fs = fs

    def get_waveform(self):
        return self._x

    def get_sampling_rate(self):
        return self._fs


def test_calculate_f0_and_accessors(monkeypatch):
    # Prepare dummy data and expected outputs
    x = np.array([0.0, 1.0, 2.0, 3.0])
    fs = 8000
    sample = DummySample(x, fs)

    expected_f0 = np.array([100.0, 200.0, 300.0])
    expected_time = np.array([0.0, 0.5, 1.0])
    expected_strength = np.array([0.1, 0.2, 0.3])

    # Capture the arguments swipep is called with
    called = {}

    def fake_swipep(x_arg, fs_arg, plim_arg, hop_arg, dlog2p_arg, dERBs_arg, sTHR_arg):
        called['args'] = (x_arg, fs_arg, plim_arg, hop_arg,
                          dlog2p_arg, dERBs_arg, sTHR_arg)
        return expected_f0, expected_time, expected_strength

    # Monkeypatch swipep inside your module
    monkeypatch.setattr(
        "VuVoPy.data.utils.fundamental_frequency.swipep",
        fake_swipep,
    )

    # Use non-default parameters
    custom_plim = (50, 400)
    custom_hop = 256
    custom_dlog2p = 1/24
    custom_dERBs = 0.05
    custom_sTHR = -2.0

    ff = FundamentalFrequency(
        sample,
        plim=custom_plim,
        hop_size=custom_hop,
        dlog2p=custom_dlog2p,
        dERBs=custom_dERBs,
        sTHR=custom_sTHR,
    )

    # Verify swipep was called exactly once with our parameters
    assert called['args'] == (
        x,
        fs,
        custom_plim,
        custom_hop,
        custom_dlog2p,
        custom_dERBs,
        custom_sTHR,
    )

    # Check that getters return what swipep returned
    np.testing.assert_array_equal(ff.get_f0(), expected_f0)
    np.testing.assert_array_equal(ff.get_time(), expected_time)
    np.testing.assert_array_equal(ff.get_strength(), expected_strength)

    # Check that sampling rate is preserved
    assert ff.get_sampling_rate() == fs


def test_default_parameters(monkeypatch):
    # When only sample is given, defaults should be passed to swipep
    x = np.zeros(10)
    fs = 16000
    sample = DummySample(x, fs)

    called = {}

    def fake_swipep(x_arg, fs_arg, plim_arg, hop_arg, dlog2p_arg, dERBs_arg, sTHR_arg):
        called['args'] = (plim_arg, hop_arg, dlog2p_arg, dERBs_arg, sTHR_arg)
        # Return empty arrays so we don't care about content here
        return np.array([]), np.array([]), np.array([])

    monkeypatch.setattr(
        "VuVoPy.data.utils.fundamental_frequency.swipep",
        fake_swipep,
    )

    ff = FundamentalFrequency(sample)  # use all defaults

    # The defaults in the signature are:
    # plim=(30,500), hop_size=512, dlog2p=1/96, dERBs=0.1, sTHR=-np.inf
    default_plim = (30, 500)
    default_hop = 512
    default_dlog2p = 1/96
    default_dERBs = 0.1
    default_sTHR = -np.inf

    # Only inspect the args after x, fs
    plim_arg, hop_arg, dlog2p_arg, dERBs_arg, sTHR_arg = called['args']
    assert plim_arg == default_plim
    assert hop_arg == default_hop
    assert dlog2p_arg == pytest.approx(default_dlog2p)
    assert dERBs_arg == pytest.approx(default_dERBs)
    assert sTHR_arg == default_sTHR
