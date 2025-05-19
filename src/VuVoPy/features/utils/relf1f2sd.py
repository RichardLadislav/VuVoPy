import numpy as np  
from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.containers.segmentation import Segmented as sg
from VuVoPy.data.utils.formant_frequencies import FormantFrequencies as ff

def relF1SD(folder_path, winlen = 512, winover = 256, wintype = 'hann'):
    """
    Compute the relative standard deviation of the first formant frequency (F1).

    This function segments a WAV audio sample, extracts the first formant (F1),
    and returns its relative standard deviation (std / mean).

    Args:
        folder_path (str): Path to the voice sample (WAV).
        winlen (int, optional): Length of the analysis window in samples. Default is 512.
        winover (int, optional): Overlap between consecutive windows in samples. Default is 256.
        wintype (str, optional): Type of window function to use. Default is 'hann'.

    Returns:
        float: Relative standard deviation of the F1 frequency.
    """
    formant_freqs = ff.from_voice_sample(sg.from_voice_sample(pp.from_voice_sample(vs.from_wav(folder_path)),winlen, wintype ,winover)).get_formants_preem()[:,0]
    if  formant_freqs.size == 0:
        return float('nan')  # or `return np.inf` for constant‐F0 case
    sigma = np.std(formant_freqs)
    if sigma == 0:
        return float('nan')
    mu  = np.mean(formant_freqs)
    return mu / sigma

def relF2SD(folder_path, winlen = 512, winover = 256, wintype = 'hann'):
    """
    Compute the relative standard deviation of the second formant frequency (F2).

    This function processes a WAV voice sample, segments it using a windowing
    approach, extracts F2 formants, and calculates the relative standard deviation
    (std/mean) of the second formant frequency.

    Args:
        folder_path (str): Path to the WAV file.
        winlen (int, optional): Window length in samples. Default is 512.
        winover (int, optional): Overlap between windows in samples. Default is 256.
        wintype (str, optional): Type of window function. Default is 'hann'.

    Returns:
        float: Relative standard deviation of F2 (std/mean).

    Notes:
        - Requires modules `vs`, `pp`, `sg`, `ff` for preprocessing and formant extraction.
        - Input file must contain a valid speech sample for accurate results.
    """


    formant_freqs = ff.from_voice_sample(sg.from_voice_sample(pp.from_voice_sample(vs.from_wav(folder_path)),winlen, wintype ,winover)).get_formants_preem()[:,1]

    if  formant_freqs.size == 0:
        return float('nan')  # or `return np.inf` for constant‐F0 case
    sigma = np.std(formant_freqs)
    if sigma == 0:
        return float('nan')
    mu  = np.mean(formant_freqs)
    return mu / sigma

def main():
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//vowel_e_test.wav"
    out = relF1SD(folder_path, wintype="hamm")
    out1 = relF2SD(folder_path)
    print(out)
    print(out1)
if __name__ == "__main__": 
    main()