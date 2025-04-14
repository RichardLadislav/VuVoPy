import numpy as np
import matplotlib.pyplot as plt
from data.containers.prepocessing import Preprocessed as pp
from data.containers.sample import VoiceSample as vs
from data.containers.segmentation import Segmented as sg
from data.utils.vuvs_gmm import vuvs_gmm
 
class Vuvs:
    """Class to detect voiced/unvoiced/scilence segments in an audio signal using GMM."""
    
    def __init__(self, segment, fs, winlen = 512, winover = 496, wintype = 'hann', smoothing_window=5):
        self.segment = segment.get_segment()
        self.segment_preem = segment.get_preem_segment()
        self.segment_norm = segment.get_norm_segment()
        self.fs = fs
        self.winlen = winlen
        self.winover = winover
        self.wintype = wintype
        self.smoothing_window = smoothing_window

        # Compute vuvs vuvs upon initialization
        self.vuvs = self.calculate_vuvs()
        
    def calculate_vuvs(self):
        """Compute voiced/unvoiced/scilence segments using GMM."""
        return vuvs_gmm(self.segment, self.fs, self.winover, self.smoothing_window)

    def get_vuvs(self):
        """Return computed voiced/unvoiced/scilence segments."""
        return self.vuvs

    def get_sampling_rate(self):
        """Return the sampling rate."""
        return self.fs
    
    def get_total_silence_duration(self, min_silence_duration_ms=50):
        """Return total duration (in seconds) of silences longer than the threshold."""
        labels = self.vuvs
        hop_duration = (self.winlen - self.winover) / self.fs
        min_frames = int(np.ceil(min_silence_duration_ms / 1000 / hop_duration))

        total_duration = 0.0
        i = 0
        while i < len(labels):
            if labels[i] == 0:
                start = i
                while i < len(labels) and labels[i] == 0:
                    i += 1
                silence_len = i - start
                if silence_len >= min_frames:
                    total_duration += silence_len * hop_duration
            else:
                i += 1
        return total_duration

    def get_silence_count(self, min_silence_duration_ms=50):
        """Return number of silent segments longer than the threshold."""
        labels = self.vuvs
        hop_duration = (self.winlen - self.winover) / self.fs
        min_frames = int(np.ceil(min_silence_duration_ms / 1000 / hop_duration))

        silence_count = 0
        i = 0
        while i < len(labels):
            if labels[i] == 0:
                start = i
                while i < len(labels) and labels[i] == 0:
                    i += 1
                silence_len = i - start
                if silence_len >= min_frames:
                    silence_count += 1
            else:
                i += 1
        return silence_count

    def get_silence_durations(self, min_silence_duration_ms=50):
        """Return list of durations (in seconds) for all silences longer than the threshold."""
        labels = self.vuvs
        hop_duration = (self.winlen - self.winover) / self.fs
        min_frames = int(np.ceil(min_silence_duration_ms / 1000 / hop_duration))

        durations = []
        i = 0
        while i < len(labels):
            if labels[i] == 0:
                start = i
                while i < len(labels) and labels[i] == 0:
                    i += 1
                silence_len = i - start
                if silence_len >= min_frames:
                    durations.append(silence_len * hop_duration)
            else:
                i += 1
        return durations

    

def main():
    """Main function to demonstrate the usage of Vuvs class."""
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//activity_unproductive.wav"
    vsample = vs.from_wav(folder_path)
    preprocessed_sample = pp.from_voice_sample(vsample)
    segment = sg.from_voice_sample(preprocessed_sample, winlen=512, wintype='hamm', winover=496, alpha=0.94)
    vuvs = Vuvs(segment, fs=vsample.get_sampling_rate(), winlen =segment.get_window_length(), winover = segment.get_window_overlap(), wintype=segment.get_window_type(), smoothing_window=5)
    y = vsample.get_waveform()
    labels = vuvs.get_vuvs()
    sr = vsample.get_sampling_rate()
    hop_length = segment.get_window_length() - segment.get_window_overlap()
    time = np.linspace(0, len(y) / sr, num=len(y))

    class_line = np.array([(-1 if l == 0 else 0 if l == 1 else 1) for l in labels])

    # Stretch classification line to match waveform length
    class_signal = np.zeros_like(y, dtype=float)
    for i, value in enumerate(class_line):
        start = i * hop_length
        end = start + hop_length
        class_signal[start:end] = value
 
    plt.figure(figsize=(14, 5))
    plt.plot(time, y / np.max(np.abs(y)), label="Normalized Waveform", color='gray', alpha=0.6)
    plt.plot(time, class_signal, label="V/UV/S Classification", color='black', linewidth=1.5)

    plt.title("Speech Waveform with Voiced / Unvoiced / Silence Classification")
    plt.xlabel("Time [s]")
    plt.yticks([-1, 0, 1], ["Silence", "Unvoiced", "Voiced"])
    plt.ylim(-1.5, 1.5)
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show(block= True)
        
if __name__ == "__main__":
    main()
    