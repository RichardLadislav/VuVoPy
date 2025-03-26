import numpy as np
#import VoiceSample

class Preemphasis(object):
    """Pre-emphasis filter applied to a VoiceSample."""

    def __init__(self, x: np.ndarray, fs: int, alpha=0.94):
        """Apply pre-emphasis and store result."""
        x_preem = np.append(x[0], x[1:] - alpha * x[:-1])
        super().__init__(x_preem, fs)  # Initialize parent class

    @classmethod
    def from_voice_sample(cls, voice_sample, alpha=0.94):
        """Create a Preemphasis object from a VoiceSample."""
        return cls(voice_sample.get_waveform(), voice_sample.get_sampling_rate(), alpha)


class SignalNormalization(object):
    """Normalize a VoiceSample."""

    def __init__(self, x: np.ndarray, fs: int):
        """Normalize the waveform and store the result."""
        x_norm = x / np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else x
        super().__init__(x_norm, fs)  # Initialize parent class

    @classmethod
    def from_voice_sample(cls, voice_sample):
        """Create a normalized VoiceSample from an existing one."""
        return cls(voice_sample.get_waveform(), voice_sample.get_sampling_rate())

    #TODO: Add segmentation function 
    #TODO: Add denoising function 