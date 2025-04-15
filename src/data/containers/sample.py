import librosa
import numpy as np
import matplotlib.pyplot as plt

class VoiceSample:
    """Class to load and process audio samples."""

    def __init__(self, x: np.ndarray, fs: int):
        """Initialize with audio waveform and sampling rate."""
        self.x = x
        self.fs = fs

    @classmethod
    def from_wav(cls, file_path: str, sr: int = None):
        """Load a WAV file and return a VoiceSample instance."""
        x, fs = librosa.load(file_path, sr=sr)
        return cls(x, fs)

    def get_waveform(self):
        """Return the waveform as a NumPy array."""
        return self.x

    def get_sampling_rate(self):
        """Return the sampling rate."""
        return self.fs

def main():
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//vowel_e_test.wav"
    aud = VoiceSample.from_wav(folder_path)
    x = aud.get_waveform()
    fs = aud.get_sampling_rate()
    print(fs)
    plt.plot(x)