import numpy as np  
from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.containers.segmentation import Segmented as sg
from VuVoPy.data.utils.vuvs_detection import Vuvs as vuvs

def ppr(folder_path, winlen = 512, winover = 496 , wintype = 'hamm', min_silence_duration_ms= 100):
    """
    Compute the percentage of silence in an audio file using voice activity detection.

    This function loads and preprocesses an audio file, segments it using a specified windowing 
    approach, and applies a voiced/unvoiced/silence detection algorithm to estimate 
    the percentage of silence.

    Args:
        folder_path (str): Path to the audio file (e.g., WAV format).
        winlen (int, optional): Window length for segmentation. Defaults to 512.
        winover (int, optional): Overlap between consecutive windows. Defaults to 496.
        wintype (str, optional): Window type ('hann', 'hamm', 'blackman', 'square'). Defaults to 'hamm'.
        min_silence_duration_ms (int, optional): Minimum duration of silence to count, in milliseconds. Defaults to 100.

    Returns:
        float: Percentage of silence in the audio signal.

    Notes:
        - Ensure the input file is in a compatible format (e.g., mono WAV).
        - The silence detection accuracy depends on the quality of preprocessing and the VAD algorithm.
    """
    # Load and preprocess the audio sample
    preprocessed_sample = pp.from_voice_sample(vs.from_wav(folder_path))
    # Segment the preprocessed sample
    segment = sg.from_voice_sample(preprocessed_sample, winlen, wintype, winover)
    fs = segment.get_sampling_rate()
    hop_size = segment.get_window_length() - segment.get_window_overlap()
    labels = vuvs(segment, fs=fs, winlen =segment.get_window_length(), winover = segment.get_window_overlap(), wintype=segment.get_window_type(), smoothing_window=5)
    
    return labels.get_total_silence_duration(min_silence_duration_ms=min_silence_duration_ms) / (len(preprocessed_sample.get_waveform())/fs) * 100

def main():
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//activity_unproductive.wav"
    out = ppr(folder_path)
    print(out)

if __name__ == "__main__":
    main()