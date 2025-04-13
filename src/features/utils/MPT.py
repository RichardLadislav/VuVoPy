import numpy as np  
from data.containers.prepocessing import Preprocessed as pp
from data.containers.sample import VoiceSample as vs
from data.containers.segmentation import Segmented as sg
from data.utils.vuvs_detection import Vuvs as vuvs
def mpt(folder_path, winlen = 512, winover = 496 , wintype = 'hamm'):
    """
    Calculate the mean pitch period of the audio signal.
    
    Parameters:
    - folder_path : str : Path to the audio file.
    - winlen      : int : Length of the window for segmentation.
    - winover     : int : Overlap between consecutive windows.
    - wintype     : str : Type of windowing function ('hann', 'hamm', 'blackman', 'square').
    
    Returns:
    - float : Mean pitch period of the audio signal.
    """
    # Load and preprocess the audio sample
    preprocessed_sample = pp.from_voice_sample(vs.from_wav(folder_path))
    # Segment the preprocessed sample
    segment = sg.from_voice_sample(preprocessed_sample, winlen, wintype, winover)
    fs = segment.get_sampling_rate()
    hop_size = segment.get_window_length() - segment.get_window_overlap()
    lables = vuvs(segment, fs=fs, winlen =segment.get_window_length(), winover = segment.get_window_overlap(), wintype=segment.get_window_type(), smoothing_window=5).get_vuvs()

    return (np.sum(lables==2)*hop_size)/fs

def main():
    folder_path = "file_path_here"
    out = mpt(folder_path)
    print(out)

if __name__ == "__main__": 
    main()
    
    