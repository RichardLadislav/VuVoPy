import numpy as np
import matplotlib.pyplot as plt
from data.containers.prepocessing import Preprocessed 
from data.containers.sample import VoiceSample 
from data.utils.swipep import swipep

class FundamentalFrequency:
    """Class to compute F0 pitch using the SWIPE' algorithm."""

    def __init__(self, sample, plim=(30, 500), hop_size = 512, dlog2p=1/96, dERBs=0.1, sTHR=-np.inf):
        """
        Initialize with a VoiceSample, Preemphasis, or SignalNormalization object.
`
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
    
def main():
    """Main function to demonstrate the usage of FundamentalFrequency class."""
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//vowel_e_test.wav"
    ff = FundamentalFrequency(VoiceSample.from_wav(folder_path),hop_size=127)
    f0 = ff.get_f0()
    time = ff.get_time()
    strength = ff.get_strength()

        # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(time, f0, label='F0', marker="x")
    plt.title("Fundamental Frequency Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid()
    plt.show(block = True)

if __name__ == "__main__":
    main()
