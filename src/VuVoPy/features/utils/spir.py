from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.containers.segmentation import Segmented as sg
from VuVoPy.data.utils.vuvs_detection import Vuvs as vuvs

def spir(folder_path, winlen = 512, winover = 496 , wintype = 'hamm'):
    """
    Calculate the percentage of silence in an audio signal using a windowing approach.

    This function segments the signal using a specified window type and size, detects
    silent segments, and returns the ratio of silence duration to the total duration
    as a percentage.

    Args:
        folder_path (str): Path to the WAV audio file.
        winlen (int, optional): Window length for segmentation. Default is 512.
        winover (int, optional): Overlap between consecutive windows. Default is 496.
        wintype (str, optional): Type of window function to use. Options are:
            'hann', 'hamm', 'blackman', 'square'. Default is 'hamm'.

    Returns:
        float: Percentage of silence in the signal.
    """
    
    preprocessed_sample = pp.from_voice_sample(vs.from_wav(folder_path))
    segment = sg.from_voice_sample(preprocessed_sample, winlen, wintype, winover)
    fs = segment.get_sampling_rate()
    labels = vuvs(segment, fs=fs, winlen =segment.get_window_length(), winover = segment.get_window_overlap(), wintype=segment.get_window_type(), smoothing_window=5)
    
    return labels.get_silence_count() / (len(preprocessed_sample.get_waveform())/fs) 