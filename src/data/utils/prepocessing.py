import numpy as np


class Preemphasis(object):
    """Class implementing preemphasis"""

    def __init__(self, xpreem: np.ndarray, fs: int):
        self.xpreem = xpreem
        self. fs = fs
    @classmethod
    def calculate_preemphasis(cls, voice_sampe, alpha = 0.94):
         """Apply pre-emphasis to a VoiceSample instance."""
         x= voice_sampe.get_waveform()
         fs= voice_sampe.get_sampling_rate()

         xpreem = np.append(x[0], x[1:] - alpha[:-1])
         return cls(xpreem)


class SignalNormalization(object):
    """Class implementing signal normalization"""

    def __init__(self, xnorm: np.ndarray, fs: int):
        self.xnorm = xnorm
        self.fs = fs
    @classmethod
    def calculate_normalization(cls, voice_sample):
        """Normalize a VoiceSample instance to the range [-1, 1]."""
        x = voice_sample.get_waveform()
        fs = voice_sample.get_sampling_rate()
        # Normalize signal: x_norm = x / max(|x|)
        xnorm = x / np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else x

        return cls(xnorm, fs)