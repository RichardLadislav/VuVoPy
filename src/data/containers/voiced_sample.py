import numpy as np
import matplotlib.pyplot as plt
from data.containers.prepocessing import Preprocessed as pp
from data.containers.sample import VoiceSample as vs
from data.containers.segmentation import Segmented as sg
from data.utils.vuvs_detection import Vuvs as vuvs

class VoicedSample:
    """
    Class to detect voiced/unvoiced/silence segments in an audio signal using GMM.
    """
    
    def __init__(self, preprocessed, vuvs, fs) :
        self.x = preprocessed.get_waveform()
        self.x_preem = preprocessed.get_preemphasis()
        self.x_norm = preprocessed.get_normalization()
        self.fs = preprocessed.get_sampling_rate()
        self.vuvs = vuvs


    def label_stretch(self):
        """
        Stretch the labels to match signal length.
        """

        labels = self.vuvs.get_vuvs()
        arr = np.asarray(labels)
        target_len = len(self.x)
        # Find segments where values stay the same
        segments = []
        start_idx = 0
        for i in range(1, len(arr)):
           if arr[i] != arr[i - 1]:
              segments.append(arr[start_idx:i])
              start_idx = i
        segments.append(arr[start_idx:])  # Add last segment

        # Determine how many samples per segment
        original_lens = np.array([len(seg) for seg in segments])
        total_original = np.sum(original_lens)
    
        # Calculate how much to stretch each segment
        stretched_lens = np.round((original_lens / total_original) * target_len).astype(int)

        # Fix rounding errors to exactly match target_len
        diff = target_len - np.sum(stretched_lens)
        while diff != 0:
            for i in range(len(stretched_lens)):
               if diff == 0:
                  break
            stretched_lens[i] += 1 if diff > 0 else -1
            diff = target_len - np.sum(stretched_lens)

        # Build the stretched array
        stretched = np.concatenate([np.full(l, seg[0]) for seg, l in zip(segments, stretched_lens)])
        return stretched
    
    def get_voiced_sample(self):
        """
        Return the voiced sample.
        """
        #time_vuvs, labels = self.vuvs.get_timed_vuvs()
        sample = self.x
        labels = self.label_stretch()
        voiced_sample = sample[labels == 2]
        return voiced_sample

    def get_silence_remove_sample(self):
        """
        Return the silence removed sample.
        """     
        sample = self.x
        labels = self.label_stretch()
        
        i = 0
        min_frames = int(np.ceil(50 / 1000 * self.fs))
        silence_idx = []

        while i < len(labels):
            if labels[i] == 0:
                start = i
                while i < len(labels) and labels[i] == 0:
                    i += 1
                silence_len = i - start
                if silence_len >= min_frames:
                    silence_idx.append((start, i))
            else:
                i += 1
            mask = np.ones(len(self.x), dtype=bool)
            for start, end in silence_idx:
                mask[start:end] = False
        
        return self.x[mask]
    
def main():
    folder_path = "file_path.wav"
    preprocessed_sample = pp.from_voice_sample(vs.from_wav(folder_path))
    segment = sg.from_voice_sample(preprocessed_sample, winlen=512, winover=496, wintype='hamm')
    fs = segment.get_sampling_rate()
    
    labels = vuvs(segment, fs=fs, winlen=segment.get_window_length(), winover=segment.get_window_overlap(), wintype=segment.get_window_type(), smoothing_window=5)
    voiced_sample = VoicedSample(preprocessed_sample,labels,fs).get_voiced_sample()
    silence_removed_sample = VoicedSample(preprocessed_sample, labels, fs).get_silence_remove_sample()
    voiced_sample = VoicedSample(preprocessed_sample, labels, fs).get_voiced_sample()
    stretched_labels = VoicedSample(preprocessed_sample, labels, fs).label_stretch()
    #    Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(stretched_labels, label='stretched labels')
    plt.figure(figsize=(12, 6))
    plt.plot(labels.get_vuvs(), label='origianl labels')

    plt.figure(figsize=(12, 6))
    plt.plot(voiced_sample, label='voiced signal')
    plt.figure(figsize=(12, 6))
    plt.plot(preprocessed_sample.get_waveform(), label='voiced signal')
    plt.figure(figsize=(12, 6))
    plt.plot(silence_removed_sample, label='voiced signal')
if __name__ == "__main__":
    main()