import numpy as np
from ...data.containers.prepocessing import Preprocessed as pp
from ...data.containers.sample import VoiceSample as vs
from ...data.containers.segmentation import Segmented as sg
from ...data.utils.vuvs_detection import Vuvs as vuvs

def durmad(folder_path, winlen=512, winover=496, wintype='hamm'):
    """
    Compute the absolute median deviation of silence durations from a voice sample.

    This function processes a voice sample from a given folder path, segments it using
    the specified window parameters, and calculates the silence durations. It then
    returns the absolute median deviation of these silence durations.

    Args:
        folder_path (str): Path to the folder containing the WAV voice sample.
        winlen (int, optional): Length of the analysis window. Default is 512.
        winover (int, optional): Overlap between windows. Default is 496.
        wintype (str, optional): Type of windowing function (e.g., 'hamm' for Hamming). Default is 'hamm'.

    Returns:
        float: Absolute median deviation of silence durations in seconds.
    """
    # Preprocess the voice sample
    preprocessed_sample = pp.from_voice_sample(vs.from_wav(folder_path))
    segment = sg.from_voice_sample(preprocessed_sample, winlen, wintype, winover)
    fs = segment.get_sampling_rate()

    # Perform voiced/unvoiced detection
    labels = vuvs(
        segment,
        fs=fs,
        winlen=segment.get_window_length(),
        winover=segment.get_window_overlap(),
        wintype=segment.get_window_type(),
        smoothing_window=5
    )

    # Get silence durations
    silence_dur = labels.get_silence_durations()

    # Handle the case where there are no silence durations
    if len(silence_dur) == 0:  # Correctly check if the array is empty
        return np.nan

    # Calculate the mean absolute deviation from the median
    return np.mean(np.abs(silence_dur - np.median(silence_dur)))