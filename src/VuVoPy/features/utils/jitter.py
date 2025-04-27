import numpy as np  
from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.utils.fundamental_frequency import FundamentalFrequency as f0
from VuVoPy.data.containers.segmentation import Segmented as sg
from VuVoPy.data.utils.vuvs_detection import Vuvs as vuvs
from VuVoPy.data.containers.voiced_sample import VoicedSample as vos

def jiterPPQ(folder_path, n_points = 3, plim=(30, 500), hop_size = 512, dlog2p=1/96, dERBs=0.1, sTHR=-np.inf):
    """
    Calculate the Pitch Perturbation Quotient (PPQ) jitter for a given audio file.
    This function computes the jitter based on the fundamental frequency (F0) 
    extracted from the audio file. Jitter is a measure of the frequency variation 
    in the voice signal.
    Parameters:
        folder_path (str): Path to the audio file (in WAV format) to analyze.
        n_points (int, optional): Number of points to consider for calculating 
            the average F0. Default is 3.
        plim (tuple, optional): Tuple specifying the pitch range (in Hz) for 
            F0 extraction. Default is (30, 500).
        hop_size (int, optional): Hop size for F0 extraction. Default is 512.
        dlog2p (float, optional): Logarithmic step size for pitch analysis. 
            Default is 1/96.
        dERBs (float, optional): Step size in Equivalent Rectangular Bandwidths 
            for pitch analysis. Default is 0.1.
        sTHR (float, optional): Silence threshold for F0 extraction. Default is -np.inf.
    Returns:
        float: The average PPQ jitter value. If the number of F0 points is less 
        than `n_points`, the function returns 0.
    """
    
    fundamental_freq = f0(vs.from_wav(folder_path), plim, hop_size, dlog2p, dERBs, sTHR).get_f0()
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
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//activity_unproductive.wav"
    out = jiterPPQ(folder_path, n_points=3, sTHR= 0.5)
    print(out)
if __name__ == "__main__": 
    main()