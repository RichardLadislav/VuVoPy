import numpy as np
from data.containers.prepocessing import Preprocessed as pp
from data.containers.sample import VoiceSample as vs
from data.containers.segmentation import Segmented as sg
from data.utils.fundamental_frequency import FundamentalFrequency as ff

def hnr(folder_path, plim=(30, 500), hop_size = 512, dlog2p=1/96, dERBs=0.1, sTHR=-np.inf):
    """
    Compute Harmonics-to-Noise Ratio (HNR) using the FundamentalFrequency class for F0 estimation.

    Parameters:
    - folder_path: Path to the audio file  
    - winlen: Frame length in frames (default: 512)
    - winover: Overlap in frames (default: 256)
    - wintype: Window type (default: "hann")
    - f0_min: Minimum fundamental frequency (Hz)
    - f0_max: Maximum fundamental frequency (Hz)

    Returns:
    - Mean HNR value across frames
    """

    # Load and preprocess the audio file
    voice_sample = vs.from_wav(folder_path)
    processed_sample = pp.from_voice_sample(voice_sample)
    
    fs = voice_sample.get_sampling_rate()  # Get sampling rate

    # Compute fundamental frequency using FundamentalFrequency class
    fundamental_freq = ff(vs.from_wav(folder_path), plim, hop_size, dlog2p, dERBs, sTHR).get_f0()

    fundamental_freq_1 = fundamental_freq[np.nonzero(fundamental_freq>40)]  # Remove zeros and values below 30 hz
    hnr_values = []
    for f0 in fundamental_freq_1:
        if np.isnan(f0) or f0 <= 0:
            continue

        # Compute harmonic-to-noise approximation
        r_max = np.exp(-f0 / (fs / 2))  # Approximate harmonicity measure
        hnr = 10 * np.log10(r_max / (1 - r_max)) if 0 < r_max < 1 else np.nan
        hnr_values.append(hnr)

    return np.nanmean(hnr_values) if len(hnr_values) > 0 else float('nan')

if __name__ == "__main__":
    file_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//vowel_e_test.wav"

    # Compute HNR
    hnr_value = hnr(file_path)

    if np.isnan(hnr_value):
        print("Could not compute HNR.")
    else:
        print(f"Mean HNR: {hnr_value:.2f} dB")
