import numpy as np  
from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.containers.segmentation import Segmented as sg
from VuVoPy.data.utils.vuvs_detection import Vuvs as vuvs

def duv(folder_path, winlen = 512, winover = 496 , wintype = 'hamm'):
    """
    Computes the voiced-to-unvoiced ratio (VUV) percentage for an audio sample.
    This function processes an audio file located at the specified folder path,
    segments it using the given window parameters, and calculates the percentage
    of voiced frames in the sample.
    Parameters:
        folder_path (str): The path to the folder containing the audio file in WAV format.
        winlen (int, optional): The length of the analysis window in samples. Default is 512.
        winover (int, optional): The overlap between consecutive windows in samples. Default is 496.
        wintype (str, optional): The type of window function to apply (e.g., 'hamm' for Hamming). Default is 'hamm'.
    Returns:
        float: The percentage of voiced frames in the audio sample.
    Notes:
        - The function assumes the presence of preprocessing, segmentation, and VUV classification
          utilities (`pp`, `vs`, `sg`, and `vuvs`) in the codebase.
        - The `smoothing_window` parameter for VUV classification is set to 5 by default.
    """
    
    # Load and preprocess the audio sample
    preprocessed_sample = pp.from_voice_sample(vs.from_wav(folder_path))
    # Segment the preprocessed sample
    segment = sg.from_voice_sample(preprocessed_sample, winlen, wintype, winover)
    fs = segment.get_sampling_rate()
    labels = vuvs(segment, fs=fs, winlen =segment.get_window_length(), winover = segment.get_window_overlap(), wintype=segment.get_window_type(), smoothing_window=5).get_vuvs()

    return np.sum(labels==1)/len(labels) * 100
