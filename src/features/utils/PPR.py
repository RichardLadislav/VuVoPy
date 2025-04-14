import numpy as np  
from data.containers.prepocessing import Preprocessed as pp
from data.containers.sample import VoiceSample as vs
from data.containers.segmentation import Segmented as sg
from data.utils.vuvs_detection import Vuvs as vuvs

def ppr(folder_path, winlen = 512, winover = 496 , wintype = 'hamm'):
    """
    Calculate the percentage of silence in an audio signal.
    This function processes an audio file to determine the proportion of    
    silence in the signal. It uses a windowing approach to segment the audio
    
    Parameters:
    - folder_path : str : Path to the audio file.
    - winlen      : int : Length of the window for segmentation.
    - winover     : int : Overlap between consecutive windows.
    - wintype     : str : Type of windowing function ('hann', 'hamm', 'blackman', 'square').
    
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