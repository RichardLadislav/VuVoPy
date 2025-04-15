import numpy as np
import librosa as lib
from data.containers.prepocessing import Preprocessed as pp
from data.containers.sample import VoiceSample as vs
from data.containers.segmentation import Segmented as sg
from data.utils.vuvs_detection import Vuvs as vuvs
from data.containers.voiced_sample import VoicedSample as vos

def relSEOSD(folder_path, winlen = 512, winover = 496 , wintype = 'hamm'):
    """
    Computes the relative standard deviation of the root mean square (RMS) energy 
    contour of a voice sample after silence removal.
    Parameters:
    -----------
        folder_path : str
            Path to the folder containing the voice sample in WAV format.
        winlen : int, optional
            Length of the analysis window in samples. Default is 512.
        winover : int, optional
            Overlap between consecutive windows in samples. Default is 496.
        wintype : str, optional
            Type of window function to apply (e.g., 'hamm' for Hamming window). Default is 'hamm'.
    Returns:
    --------
    float
        The relative standard deviation (standard deviation divided by the mean) 
        of the RMS energy contour. Returns 0 if the mean RMS energy is zero.
    """


    preprocessed_sample = pp.from_voice_sample(vs.from_wav(folder_path))
    segment = sg.from_voice_sample(preprocessed_sample, winlen, wintype, winover)
    fs = segment.get_sampling_rate()
    labels = vuvs(segment, fs=fs, winlen =segment.get_window_length(), winover = segment.get_window_overlap(), wintype=segment.get_window_type(), smoothing_window=5)
    silence_removed_sample = vos(preprocessed_sample, labels, fs).get_silence_remove_sample()
    contour = lib.feature.rms(y=silence_removed_sample, frame_length=winlen, hop_length=(winlen-winover), center=False)
    mean_val = np.mean(contour)
    std_val = np.std(contour)

    return std_val / mean_val if mean_val != 0 else 0
def main():
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//activity_unproductive.wav"
    out = relSEOSD(folder_path)
    print(f"Relative standard deviation of RMS energy contour: {out}")

if __name__ == "__main__":
    main()