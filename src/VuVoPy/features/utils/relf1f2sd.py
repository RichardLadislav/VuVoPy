import numpy as np  
from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.containers.segmentation import Segmented as sg
from VuVoPy.data.utils.formant_frequencies import FormantFrequencies as ff

def relF1SD(folder_path, winlen = 512, winover = 256, wintype = 'hann'):
    """
    Computes the relative standard deviation of the first formant frequency (F1) 
    from a voice sample.

    Parameters:
    -----------
    folder_path : str
        Path to the folder containing the voice sample in WAV format.
    winlen : int, optional
        Length of the analysis window in samples (default is 512).
    winover : int, optional
        Overlap between consecutive windows in samples (default is 256).
    wintype : str, optional
        Type of the window function to apply (default is 'hann').

    Returns:
    --------
    float
        The relative standard deviation of the first formant frequency (F1).
    """
    formant_freqs = ff.from_voice_sample(sg.from_voice_sample(pp.from_voice_sample(vs.from_wav(folder_path)),winlen, wintype ,winover))
    return np.mean(formant_freqs.get_formants_preem()[:,0])/np.std(formant_freqs.get_formants_preem()[:,0])

def relF2SD(folder_path, winlen = 512, winover = 256, wintype = 'hann'):
    """
    Computes the relative standard deviation of the second formant frequency (F2) 
    from a voice sample provided as a WAV file.
    Parameters:
    -----------
    folder_path : str
        The file path to the WAV file containing the voice sample.
    winlen : int, optional
        The length of the analysis window in samples. Default is 512.
    winover : int, optional
        The overlap between consecutive windows in samples. Default is 256.
    wintype : str, optional
        The type of window function to apply. Default is 'hann'.
    Returns:
    --------
    float
        The relative standard deviation of the second formant frequency (F2), 
        calculated as the mean of F2 divided by its standard deviation.
    Notes:
    ------
    - This function relies on several external modules (`vs`, `pp`, `sg`, `ff`) 
      tracting formant frequencies.
    - Ensure that the input WAV file is valid and contains a proper voice sample 
      for accurate results.
    """

    formant_freqs = ff.from_voice_sample(sg.from_voice_sample(pp.from_voice_sample(vs.from_wav(folder_path)),winlen, wintype ,winover))
    return np.mean(formant_freqs.get_formants_preem()[:,1])/np.std(formant_freqs.get_formants_preem()[:,1])

def main():
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//vowel_e_test.wav"
    out = relF1SD(folder_path, wintype="hamm")
    out1 = relF2SD(folder_path)
    print(out)
    print(out1)
if __name__ == "__main__": 
    main()