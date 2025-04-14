from data.containers.prepocessing import Preprocessed as pp
from data.containers.sample import VoiceSample as vs
from data.containers.segmentation import Segmented as sg
from data.utils.vuvs_detection import Vuvs as vuvs

def spir(folder_path, winlen = 512, winover = 496 , wintype = 'hamm'):
    """
        Calculate the percentage of silence in an audio signal using a windowing approach.
        Parameters:
        - folder_path : str
            Path to the folder containing the audio file in WAV format.
        - winlen : int, optional
            Length of the analysis window (default is 512).
        - winover : int, optional
            Overlap between consecutive windows (default is 496).
        - wintype : str, optional
            Type of windowing function to apply. Options are:
            'hann', 'hamm', 'blackman', 'square' (default is 'hamm').
        - float
            The percentage of silence in the audio signal, calculated as the ratio
            of silent segments to the total duration of the signal.
    """
    
    preprocessed_sample = pp.from_voice_sample(vs.from_wav(folder_path))
    segment = sg.from_voice_sample(preprocessed_sample, winlen, wintype, winover)
    fs = segment.get_sampling_rate()
    labels = vuvs(segment, fs=fs, winlen =segment.get_window_length(), winover = segment.get_window_overlap(), wintype=segment.get_window_type(), smoothing_window=5)
    
    return labels.get_silence_count() / (len(preprocessed_sample.get_waveform())/fs) 
def main():
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//activity_unproductive.wav"
    out = spir(folder_path)
    print(out)

if __name__ == "__main__":
    main()