import numpy as np  
from data.containers.prepocessing import Preprocessed as pp
from data.containers.sample import VoiceSample as vs
from data.utils.fundamental_frequency import FundamentalFrequency as f0

def jiterPPQ(folder_path, n_points = 3, plim=(30, 500), hop_size = 512, dlog2p=1/96, dERBs=0.1, sTHR=-np.inf):
    
    fundamental_freq = f0(vs.from_wav(folder_path), plim, hop_size, dlog2p, dERBs, sTHR).get_f0()
    if len(fundamental_freq) < n_points:
        return 0
    # Create an array to hold the APQ values
    jitter_values = []
    for i in range(len(fundamental_freq) - n_points):
        avg_f0 = np.mean(fundamental_freq[i:i+n_points])  # Average of the n_points f0 values
        jitter_values.append(np.abs(fundamental_freq[i+n_points] - avg_f0))  # Jitter is the difference from the average
    return np.mean(jitter_values)

def main():
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//vowel_e_test.wav"
    out = jiterPPQ(folder_path, n_points=3)
    print(out)
if __name__ == "__main__": 
    main()