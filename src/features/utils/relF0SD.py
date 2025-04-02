import numpy as np  
from data.containers.prepocessing import Preprocessed as pp
from data.containers.sample import VoiceSample as vs
from data.utils.fundamental_frequency import FundamentalFrequency as f0

def relF0SD(folder_path, plim=(30, 500), hop_size = 512, dlog2p=1/96, dERBs=0.1, sTHR=-np.inf):
    """
    Calculate the relative standard deviation of the first formant frequency.

    Parameters:
    - folder_path : str : Path to the audio file.
    - plim        : tuple : Tuple (min_freq, max_freq) for pitch search range.
    - hop_size    : int   : Time step for analysis (seconds).
    - dlog2p      : float : Resolution of pitch candidates.
    - dERBs       : float : Frequency resolution in ERBs.
    - sTHR        : float : Pitch strength threshold.

    Returns:
    - float : Relative standard deviation of the fundamental frequency.
    """
    fundamental_freq = f0(vs.from_wav(folder_path), plim, hop_size, dlog2p, dERBs, sTHR)
    return np.mean(fundamental_freq.get_f0())/np.std(fundamental_freq.get_f0())
def main():
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//vowel_e_test.wav"
    out = relF0SD(folder_path)
    print(out)
if __name__ == "__main__": 
    main()