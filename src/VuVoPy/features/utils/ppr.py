import numpy as np  
from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.containers.segmentation import Segmented as sg
from VuVoPy.data.utils.vuvs_detection import Vuvs as vuvs

def ppr(folder_path, winlen = 512, winover = 496 , wintype = 'hamm'):
    """
    This function processes an audio file to determine the proportion of silence in the signal. 
    It uses a windowing approach to segment the audio and applies voice activity detection 
    to classify segments as silent or non-silent.
    - folder_path : str
        Path to the audio file.
    - winlen : int, optional, default=512
        Length of the window for segmentation.
    - winover : int, optional, default=496
        Overlap between consecutive windows.
    - wintype : str, optional, default='hamm'
        Type of windowing function ('hann', 'hamm', 'blackman', 'square').
    - float
        Percentage of silence in the audio signal.
    Notes:
    - The function relies on external modules for preprocessing, segmentation, and voice activity detection.
    - Ensure the input audio file is in a compatible format (e.g., WAV).
    - The accuracy of the silence percentage depends on the quality of the voice activity detection algorithm.
    Calculate the percentage of silence in an audio signal.
    This function processes an audio file to determine the proportion of    
    silence in the signal. It uses a windowing approach to segment the audio
    
    Returns:
    - float : Percentage of silence in the audio signal.
    """
    # Load and preprocess the audio sample
    preprocessed_sample = pp.from_voice_sample(vs.from_wav(folder_path))
    # Segment the preprocessed sample
    segment = sg.from_voice_sample(preprocessed_sample, winlen, wintype, winover)
    fs = segment.get_sampling_rate()
    hop_size = segment.get_window_length() - segment.get_window_overlap()
    labels = vuvs(segment, fs=fs, winlen =segment.get_window_length(), winover = segment.get_window_overlap(), wintype=segment.get_window_type(), smoothing_window=5)
    
    return labels.get_total_silence_duration() / (len(preprocessed_sample.get_waveform())/fs) * 100

def main():
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//activity_unproductive.wav"
    out = ppr(folder_path)
    print(out)

if __name__ == "__main__":
    main()