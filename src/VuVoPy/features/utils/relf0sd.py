import numpy as np  
from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.utils.fundamental_frequency import FundamentalFrequency as f0

def relF0SD(folder_path, plim=(30, 500), hop_size = 512, dlog2p=1/96, dERBs=0.1, sTHR=-np.inf):
    """
    Calculate the relative standard deviation of the fundamental frequency (F0).

    This function computes the relative standard deviation (standard deviation divided by mean)
    of the fundamental frequency (F0) extracted from an audio file.

    Args:
        folder_path (str): Path to the audio file.
        plim (tuple, optional): Tuple (min_freq, max_freq) specifying pitch range in Hz. Default is (30, 500).
        hop_size (int, optional): Time step for analysis in samples. Default is 512.
        dlog2p (float, optional): Resolution of pitch candidates in log2 space. Default is 1/96.
        dERBs (float, optional): Frequency resolution in ERBs. Default is 0.1.
        sTHR (float, optional): Pitch strength threshold. Default is -np.inf.

    Returns:
        float: Relative standard deviation of the fundamental frequency (std/mean).

    Notes:
        - Requires `vs.from_wav` and `f0` functions from VuVoPy.
        - The input file must be supported by `vs.from_wav` (e.g., WAV format).
    """
    fundamental_freq = f0(vs.from_wav(folder_path), plim, hop_size, dlog2p, dERBs, sTHR).get_f0()
    if  fundamental_freq.size == 0:
        return float('nan')  # or `return np.inf` for constant‐F0 case
    sigma = np.std(fundamental_freq)
    if sigma == 0:
        return float('nan')
    mu  = np.mean(fundamental_freq)
    return mu / sigma
def main():
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//vowel_e_test.wav"
    out = relF0SD(folder_path)
    print(out)
if __name__ == "__main__": 
    main()