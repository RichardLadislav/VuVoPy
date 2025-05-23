import numpy as np  
from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.containers.segmentation import Segmented as sg
from VuVoPy.data.utils.vuvs_detection import Vuvs as vuvs

def mpt(folder_path, winlen = 512, winover = 496 , wintype = 'hamm'):
    """
    Computes the Maximal Phonation Time (MPT) from a given audio file.
    Parameters:
        folder_path (str): The file path to the audio sample in WAV format.
        winlen (int, optional): The length of the analysis window. Default is 512.
        winover (int, optional): The overlap between consecutive windows. Default is 496.
        wintype (str, optional): The type of windowing function to apply (e.g., 'hamm' for Hamming). Default is 'hamm'.
    Returns:
        float: The Maximal Phonation Time (MPT) in seconds, calculated as the total duration of voiced segments.
    """
    
    # Load and preprocess the audio sample
    preprocessed_sample = pp.from_voice_sample(vs.from_wav(folder_path))
    # Segment the preprocessed sample
    segment = sg.from_voice_sample(preprocessed_sample, winlen, wintype, winover)
    fs = segment.get_sampling_rate()
    hop_size = segment.get_window_length() - segment.get_window_overlap()
    lables = vuvs(segment, fs=fs, winlen =segment.get_window_length(), winover = segment.get_window_overlap(), wintype=segment.get_window_type(), smoothing_window=5).get_vuvs()

    return (np.sum(lables==2)*hop_size)/fs
