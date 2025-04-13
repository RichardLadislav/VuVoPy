import numpy as np  
from data.containers.prepocessing import Preprocessed as pp
from data.containers.sample import VoiceSample as vs
from data.containers.segmentation import Segmented as sg
from data.utils.vuvs_detection import Vuvs as vuvs

def duv(folder_path, winlen = 512, winover = 496 , wintype = 'hamm'):
    # Load and preprocess the audio sample
    preprocessed_sample = pp.from_voice_sample(vs.from_wav(folder_path))
    # Segment the preprocessed sample
    segment = sg.from_voice_sample(preprocessed_sample, winlen, wintype, winover)
    fs = segment.get_sampling_rate()
    labels = vuvs(segment, fs=fs, winlen =segment.get_window_length(), winover = segment.get_window_overlap(), wintype=segment.get_window_type(), smoothing_window=5).get_vuvs()

    return np.sum(labels==1)/len(labels) * 100

def main():
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//activity_unproductive.wav"
    out = duv(folder_path)
    print(out)

if __name__ == "__main__": 
    main()
    