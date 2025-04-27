import numpy as np
from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.containers.segmentation import Segmented as sg
from VuVoPy.data.utils.fundamental_frequency import FundamentalFrequency as ff

def hnr(folder_path, plim=(30, 500), hop_size = 512, dlog2p=1/96, dERBs=0.1, sTHR=-np.inf):
    """
    Compute the Harmonics-to-Noise Ratio (HNR) of an audio signal.
    HNR is a measure of the ratio of harmonic sound to noise in a signal, 
    often used in speech analysis to assess voice quality.
    Parameters:
    -----------
    folder_path : str
        Path to the audio file in WAV format.
    plim : tuple, optional
        Pitch range limits in Hz (default is (30, 500)).
    hop_size : int, optional
        Hop size for analysis in samples (default is 512).
    dlog2p : float, optional
        Logarithmic pitch step size (default is 1/96).
    dERBs : float, optional
        Step size in Equivalent Rectangular Bandwidths (default is 0.1).
    sTHR : float, optional
        Silence threshold in dB (default is -np.inf).
    Returns:
    --------
    float
        The average HNR value of the audio signal. Returns NaN if no valid 
        HNR values are computed.
    Notes:
    ------
    - The function preprocesses the audio signal and computes the fundamental 
      frequency using a pitch estimation algorithm.
    - HNR is calculated for each valid fundamental frequency value and averaged 
      across the signal.
    - Values below 40 Hz or invalid frequencies (e.g., NaN or non-positive) 
      are excluded from the computation.
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
    file_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//recordings-20250306T192251Z-001//recordings//K1019//K1019_8.2-1_1.wav"


    # Compute HNR
    hnr_value = hnr(file_path)

    if np.isnan(hnr_value):
        print("Could not compute HNR.")
    else:
        print(f"Mean HNR: {hnr_value:.2f} dB")