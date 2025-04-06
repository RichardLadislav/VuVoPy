import numpy as np
from .sample import VoiceSample
from .prepocessing import Preprocessed  # Fixed typo in import
#TODO  prepisat typo v preprocessing

class Segmented(Preprocessed):  # Fixed typo in class name
    def __init__(self, x, fs, xnorm, preem, xsegment, winlen, wintype, winover, alpha=0.94):
        super().__init__(x, fs, xnorm, preem, alpha)
        self.xsegment = xsegment if xsegment is not None else x
        self.winlen = winlen
        self.wintype = wintype
        self.winover = winover

    @classmethod
    def from_voice_sample(cls, voice_sample, winlen, wintype, winover, alpha=0.94):
        x = voice_sample.get_waveform()
        fs = voice_sample.get_sampling_rate()
        x_preem = voice_sample.get_preemphasis(alpha)
        x_norm = voice_sample.get_normalization()

        # Define window
        match wintype:
            case "hann":
                win = np.hanning(winlen)
            case "blackman":
                win = np.blackman(winlen)
            case "hamm":
                win = np.hamming(winlen)
            case "square":  # Fixed typo
                win = np.ones(winlen)
            case _:
                win = np.hamming(winlen)

        # Compute number of frames
        cols = int(np.ceil((x.size - winover) / (winlen - winover)))

        # Pad signal if necessary
        if len(x) % winlen != 0:
           x = np.pad(x, (0, cols * winlen - len(x)), mode='constant')
           x_preem = np.pad(x_preem, (0, cols * winlen - len(x_preem)), mode='constant')
           x_norm = np.pad(x_norm, (0, cols * winlen - len(x_norm)), mode='constant')

        # Initialize segmented array
        xsegment = np.zeros((winlen, cols, 3))

        # Segment
        sel = np.arange(winlen).reshape(-1, 1)
        step = np.arange(0, (cols - 1) * (winlen - winover) + 1, winlen - winover)

        xsegment[:, :, 0] = x[sel + step]  # Original waveform
        xsegment[:, :, 1] = x_preem[sel + step]  # Pre-emphasized
        xsegment[:, :, 2] = x_norm[sel + step]  # Normalized

        # Apply window
        xsegment[:, :, 0] *= win[:, np.newaxis]
        xsegment[:, :, 1] *= win[:, np.newaxis]
        xsegment[:, :, 2] *= win[:, np.newaxis]

        return cls(x, fs, x_norm, x_preem, xsegment, winlen, wintype, winover, alpha)

    def get_segment(self):
        """Return the waveform as a NumPy array."""
        return self.xsegment[:,:,0]

    def get_preem_segment(self):
        """Return the waveform as a NumPy array."""
        return self.xsegment[:,:,1]

    def get_norm_segment(self):
        """Return the waveform as a NumPy array."""
        return self.xsegment[:,:,2]

    def get_sampling_rate(self):
        """Return the sampling rate."""
        return self.fs

    def get_window_type(self):
        """Return the window type."""
        return self.wintype

    def get_window_length(self):
        """Return the window length."""
        return self.winlen

    def get_window_overlap(self):
        """Return the window overlap."""
        return self.winover

def main():
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//vowel_e_test.wav"
    processed_sample = Preprocessed.from_voice_sample(VoiceSample.from_wav(folder_path))

    seg = Segmented.from_voice_sample(processed_sample, winlen=512, winover=256, wintype="hann")

    # Apply different windowing functions dynamically
    seg_wave = seg.get_segment(winlen=512, wintype="hann", winover=256)
    seg_preem = seg.get_preem_segment(winlen=512, wintype="blackman", winover=256)
    seg_norm = seg.get_norm_segment(winlen=512, wintype="square", winover=256)
    print("holap")
if __name__ == "__main__": 
    main()