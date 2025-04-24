import numpy as np
from .sample import VoiceSample
from .prepocessing import Preprocessed  

class Segmented(Preprocessed):  
    class Segmented:
        """
        A class for segmenting and preprocessing audio signals.
        The `Segmented` class extends the `Preprocessed` class and provides functionality
        for segmenting audio signals into overlapping frames, applying window functions,
        and storing the segmented data in multiple forms (original, pre-emphasized, and normalized).
        Attributes:
            x (numpy.ndarray): The original waveform.
            fs (int): The sampling rate of the audio signal.
            xnorm (numpy.ndarray): The normalized waveform.
            preem (numpy.ndarray): The pre-emphasized waveform.
            xsegment (numpy.ndarray): A 3D array containing segmented data for the original,
                pre-emphasized, and normalized waveforms.
            winlen (int): The length of the window used for segmentation.
            wintype (str): The type of window function applied (e.g., "hann", "hamming").
            winover (int): The overlap between consecutive windows.
            alpha (float): The pre-emphasis coefficient (default is 0.94).
        Methods:
            from_voice_sample(voice_sample, winlen, wintype, winover, alpha=0.94):
                Class method to create a `Segmented` instance from a voice sample object.
            get_segment():
                Returns the segmented original waveform as a NumPy array.
            get_preem_segment():
                Returns the segmented pre-emphasized waveform as a NumPy array.
            get_norm_segment():
                Returns the segmented normalized waveform as a NumPy array.
            get_sampling_rate():
                Returns the sampling rate of the audio signal.
            get_window_type():
                Returns the type of window function applied.
            get_window_length():
                Returns the length of the window used for segmentation.
            get_window_overlap():
                Returns the overlap between consecutive windows.
        """
    def __init__(self, x, fs, xnorm, preem, xsegment, winlen, wintype, winover, alpha=0.94):
        super().__init__(x, fs, xnorm, preem, alpha)
        self.xsegment = xsegment if xsegment is not None else x
        self.winlen = winlen
        self.wintype = wintype
        self.winover = winover

    @classmethod
    def from_voice_sample(cls, voice_sample, winlen, wintype, winover, alpha=0.94):
        """
        Creates a segmentation object from a voice sample.
        This method processes a voice sample by segmenting it into overlapping frames,
        applying pre-emphasis, normalization, and a specified windowing function.
        Args:
            voice_sample (VoiceSample): The input voice sample object containing waveform,
                                        sampling rate, pre-emphasis, and normalization methods.
            winlen (int): The length of the window (in samples) to be applied to each frame.
            wintype (str): The type of window to apply. Supported types are:
                           "hann", "blackman", "hamm", "square". Defaults to "hamming" if unspecified.
            winover (int): The overlap (in samples) between consecutive frames.
            alpha (float, optional): The pre-emphasis coefficient. Defaults to 0.94.
        Returns:
            Segmentation: An instance of the Segmentation class containing the segmented
                          waveform, sampling rate, normalized waveform, pre-emphasized waveform,
                          and other parameters.
        Raises:
            ValueError: If an unsupported window type is specified.
        Notes:
            - The input waveform is padded with zeros if its length is not a multiple of the window length.
            - The segmentation process generates three versions of the signal: original, pre-emphasized,
              and normalized, each of which is windowed and stored in the output.
        """
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