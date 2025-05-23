import numpy as np  
from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.utils.fundamental_frequency import FundamentalFrequency as f0
from VuVoPy.data.containers.segmentation import Segmented as sg
from VuVoPy.data.utils.vuvs_detection import Vuvs as vuvs
from VuVoPy.data.containers.voiced_sample import VoicedSample as vos

def jitterPPQ(folder_path, n_points = 3, plim=(30, 500), hop_size = 512, dlog2p=1/96, dERBs=0.1, sTHR=-np.inf):
    """
    Calculate the Pitch Perturbation Quotient (PPQ) jitter for a given audio file.

    This function computes jitter based on the fundamental frequency (F0)
    extracted from a voice signal. Jitter reflects cycle-to-cycle variability
    in F0 and is useful in analyzing vocal stability.

    Args:
        folder_path (str): Path to the WAV audio file to analyze.
        n_points (int, optional): Number of points to average when calculating PPQ. Default is 3.
        plim (tuple, optional): Pitch range in Hz for F0 extraction. Default is (30, 500).
        hop_size (int, optional): Hop size for F0 tracking. Default is 512.
        dlog2p (float, optional): Log2 pitch step size. Default is 1/96.
        dERBs (float, optional): Frequency resolution in ERBs. Default is 0.1.
        sTHR (float, optional): Voicing threshold. Default is -np.inf.

    Returns:
        float: The average PPQ jitter value. Returns 0 if there are not enough F0 values.
    """
    
    fundamental_freq = f0(vs.from_wav(folder_path), plim, hop_size, dlog2p, dERBs, sTHR).get_f0()
    # Only keep non-zero (voiced) values
    fundamental_freq = fundamental_freq[fundamental_freq > 0]
    if len(fundamental_freq) < n_points:
        return 0.0

    k = n_points // 2
    T = 1 / fundamental_freq  # Convert frequency to period
    T_bar = np.mean(T)

    jitter_values = []
    for i in range(k, len(T) - k):
        local_avg = np.mean(T[i - k:i + k + 1])
        deviation = abs(T[i] - local_avg) / T_bar
        jitter_values.append(deviation)

    return float(np.mean(jitter_values))
