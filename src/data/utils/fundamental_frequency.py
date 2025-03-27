import numpy as np
import librosa
import swipep
class FundamentalFrequency:
    """Class to compute F0 pitch using the SWIPE' algorithm."""

    def __init__(self, sample, plim=(30, 500), dt=0.01, dlog2p=1/96, dERBs=0.1, sTHR=-np.inf):
        """
        Initialize with a VoiceSample, Preemphasis, or SignalNormalization object.

        Parameters:
        - sample : VoiceSample or derived class (Preemphasis, SignalNormalization)
        - plim   : Tuple (min_freq, max_freq) for pitch search range
        - dt     : Time step for analysis (seconds)
        - dlog2p : Resolution of pitch candidates
        - dERBs  : Frequency resolution in ERBs
        - sTHR   : Pitch strength threshold
        """
        if not isinstance(sample, object):
            raise TypeError("Input must be an instance of VoiceSample or its derived classes.")

        self.x = sample.get_waveform()
        self.fs = sample.get_sampling_rate()
        self.plim = plim
        self.dt = dt
        self.dlog2p = dlog2p
        self.dERBs = dERBs
        self.sTHR = sTHR

        # Compute F0 upon initialization
        self.f0, self.time, self.strength = self.calculate_f0()

    def calculate_f0(self):
        """Compute F0 using SWIPE' algorithm."""
        return swipep(self.x, self.fs, self.plim, self.dt, self.dlog2p, self.dERBs, self.sTHR)

    def get_f0(self):
        """Return computed fundamental frequency."""
        return self.f0

    def get_time(self):
        """Return time instances corresponding to F0."""
        return self.time

    def get_strength(self):
        """Return pitch strength values."""
        return self.strength
