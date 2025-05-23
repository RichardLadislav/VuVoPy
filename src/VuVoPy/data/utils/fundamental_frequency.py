import numpy as np
import matplotlib.pyplot as plt
from VuVoPy.data.utils.swipep import swipep
from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.containers.segmentation import Segmented as sg
from VuVoPy.data.utils.vuvs_detection import Vuvs as vuvs
from VuVoPy.data.containers.voiced_sample import VoicedSample as vos

class FundamentalFrequency:
    """
    FundamentalFrequency is a class for analyzing and extracting the fundamental frequency (F0) 
    from a given audio sample using the SWIPE' algorithm.
    Attributes:
        x (numpy.ndarray): The waveform of the input audio sample.
        fs (int): The sampling rate of the input audio sample.
        plim (tuple): A tuple specifying the pitch search range (min_freq, max_freq).
        hop_size (int): The time step for analysis in samples.
        dlog2p (float): The resolution of pitch candidates in log2 scale.
        dERBs (float): The frequency resolution in ERBs.
        sTHR (float): The pitch strength threshold.
        f0 (numpy.ndarray): The computed fundamental frequency values.
        time (numpy.ndarray): The time instances corresponding to the computed F0 values.
        strength (numpy.ndarray): The pitch strength values.
    Methods:
        __init__(sample, plim=(30, 500), hop_size=512, dlog2p=1/96, dERBs=0.1, sTHR=-np.inf):
            Initializes the FundamentalFrequency object with the given parameters and computes F0.
        calculate_f0():
            Computes the fundamental frequency (F0) using the SWIPE' algorithm.
        get_f0():
            Returns the computed fundamental frequency values.
        get_time():
            Returns the time instances corresponding to the computed F0 values.
        get_strength():
            Returns the pitch strength values.
        get_sampling_rate():
            Returns the sampling rate of the input audio sample.
    """

    def __init__(self, sample, plim=(30, 500), hop_size=512, dlog2p=1/96, dERBs=0.1, sTHR=-np.inf):
        """
        Initialize with a VoiceSample, Preemphasis, or SignalNormalization object.

        Parameters:
        - sample   : VoiceSample or derived class (Preemphasis, SignalNormalization)
        - plim     : Tuple (min_freq, max_freq) for pitch search range
        - hop_size : Time step for analysis (seconds)
        - dlog2p   : Resolution of pitch candidates
        - dERBs    : Frequency resolution in ERBs
        - sTHR     : Pitch strength threshold
        """
        if not isinstance(sample, object):
            raise TypeError("Input must be an instance of VoiceSample or its derived classes.")

        self.x = sample.get_waveform()
        self.fs = sample.get_sampling_rate()
        self.plim = plim
        self.hop_size = hop_size
        self.dlog2p = dlog2p
        self.dERBs = dERBs
        self.sTHR = sTHR

        # Compute F0 upon initialization
        self.f0, self.time, self.strength = self.calculate_f0()

    def calculate_f0(self):
        """Compute F0 using SWIPE' algorithm."""
        return swipep(self.x, self.fs, self.plim, self.hop_size, self.dlog2p, self.dERBs, self.sTHR)

    def get_f0(self):
        """Return computed fundamental frequency."""
        return self.f0

    def get_time(self):
        """Return time instances corresponding to F0."""
        return self.time

    def get_strength(self):
        """Return pitch strength values."""
        return self.strength
    
    def get_sampling_rate(self):
        """Return the sampling rate."""
        return self.fs
    