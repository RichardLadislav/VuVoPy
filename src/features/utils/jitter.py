import numpy as np  
from data.containers.prepocessing import Preprocessed as pp
from data.containers.sample import VoiceSample as vs
from data.utils.fundamental_frequency import FundamentalFrequency as f0
from data.containers.segmentation import Segmented as sg
from data.utils.vuvs_detection import Vuvs as vuvs
from data.containers.voiced_sample import VoicedSample as vos

def jiterPPQ(folder_path, n_points = 3, plim=(30, 500), hop_size = 512, dlog2p=1/96, dERBs=0.1, sTHR=-np.inf):
    
    #preprocessed_sample = pp.from_voice_sample(vs.from_wav(folder_path))
    #segment = sg.from_voice_sample(preprocessed_sample, winlen=512, winover=496, wintype='hamm')
    #fs = segment.get_sampling_rate()
    #labels = vuvs(segment, fs=fs, winlen =segment.get_window_length(), winover = segment.get_window_overlap(), wintype=segment.get_window_type(), smoothing_window=5)
    #silence_removed_sample = vos(preprocessed_sample, labels, fs)
    fundamental_freq = f0(vs.from_wav(folder_path), plim, hop_size, dlog2p, dERBs, sTHR).get_f0()
    #fundamental_freq = f0(silence_removed_sample, plim, hop_size, dlog2p, dERBs, sTHR).get_f0()
    #fundamental_freq_1 = fundamental_freq[np.nonzero(fundamental_freq>40)]  # Remove zeros and values below 30 hz
    if len(fundamental_freq) < n_points:
        return 0
    
    # Create an array to hold the APQ values
    jitter_values = []
    for i in range(len(fundamental_freq) - n_points):
        avg_f0 = np.mean(fundamental_freq[i:i+n_points])  # Mean F0 over n_points
        jitter = np.abs(fundamental_freq[i+n_points-1] - avg_f0) / avg_f0  # Normalize by mean F0
        jitter_values.append(jitter)

    return np.mean(jitter_values)  # Return average PPQ

def main():
    #folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//vowel_e_test.wav"
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//activity_unproductive.wav"
    out = jiterPPQ(folder_path, n_points=3, sTHR= 0.5)
    print(out)
if __name__ == "__main__": 
    main()