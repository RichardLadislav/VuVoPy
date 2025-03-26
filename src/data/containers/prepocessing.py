import numpy as np
import matplotlib.pyplot as plt
from sample import VoiceSample

class Preprocessed(VoiceSample):
    """Preprocessing applied to VoiceSample"""
    def __init__(self, x, fs, xnorm, preem, alpha=0.94):
        super().__init__(x, fs)
        self.xnorm = xnorm if xnorm is not None else x  # Default to x if not provided
        self.preem = preem if preem is not None else x  # Default to x if not provided
        self.alpha = alpha
        
    @classmethod
    def from_voice_sample(cls, voice_sample, alpha=0.94):
        """Apply normalization and pre-emphasis to a VoiceSample and return a Preprocessed object."""
        x = voice_sample.get_waveform()
        fs = voice_sample.get_sampling_rate()
        
        # Apply pre-emphasis
        x_preem = np.append(x[0], x[1:] - alpha * x[:-1])
        
       # Apply normalization
        x_norm = x / np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else x

        return cls(x, fs, xnorm=x_norm, preem=x_preem, alpha = alpha)


    def get_preemphasis(self, alpha = None):
        """Return the waveform as a NumPy array."""
        if alpha is None:
            return self.preem
        return np.append(self.x[0], self.x[1:] - alpha * self.x[:-1])

    def get_normalization(self):
        """Return the waveform as a NumPy array."""
        return self.xnorm

    def get_waveform(self):
        """Return the waveform as a NumPy array."""
        return self.x

    def get_sampling_rate(self):
        """Return the sampling rate."""
        return self.fs
    #TODO: Add segmentation function 
    #TODO: Add denoising function 
    # Assuming 'voice_sample' is an instance of VoiceSample
def main():
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//vowel_e_test.wav"
    processed_sample = Preprocessed.from_voice_sample(VoiceSample.from_wav(folder_path))

#    Get processed versions
    preemphasized = processed_sample.get_preemphasis()
    normalized = processed_sample.get_normalization()
    original = processed_sample.get_waveform()  
    sampling_rate = processed_sample.get_sampling_rate()
    #TODO Vymyslet kombinovany preprocessing 

    print(sampling_rate)
    plt.plot(original, label="Original")
    plt.plot(normalized, label="Normalized")
    plt.plot(preemphasized, label="Pre-emphasized")
    plt.legend()
    plt.show(block=True)
if __name__ == "__main__": 
    main()