import numpy as np
from data.containers.prepocessing import Preprocessed as pp
from data.containers.sample import VoiceSample as vs
from data.containers.segmentation import Segmented as sg
from data.utils.vuvs_detection import Vuvs as vuvs

def durmed(folder_path, winlen = 512, winover = 496 , wintype = 'hamm'):
    """
    Calculate the median silence duration from a voice sample.
    This function processes a voice sample from a given file path, segments it
    using specified window parameters, and calculates the median duration of
    silence segments.
    Args:
        folder_path (str): The file path to the voice sample in WAV format.
        winlen (int, optional): The length of the analysis window. Default is 512.
        winover (int, optional): The overlap between consecutive windows. Default is 496.
        wintype (str, optional): The type of window to apply (e.g., 'hamm' for Hamming). Default is 'hamm'.
    Returns:
        float: The median duration of silence segments in the voice sample.
    """
    
    preprocessed_sample = pp.from_voice_sample(vs.from_wav(folder_path))
    segment = sg.from_voice_sample(preprocessed_sample, winlen, wintype, winover)
    fs = segment.get_sampling_rate()
    labels = vuvs(segment, fs=fs, winlen =segment.get_window_length(), winover = segment.get_window_overlap(), wintype=segment.get_window_type(), smoothing_window=5)
    
    return np.median(labels.get_silence_durations())

def main():
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//activity_unproductive.wav"
    out = durmed(folder_path)
    print(f"Median {out}s")

if __name__ == "__main__":
    main()