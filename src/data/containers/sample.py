import librosa
import numpy as np

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
