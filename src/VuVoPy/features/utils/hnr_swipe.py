import numpy as np
from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.containers.segmentation import Segmented as sg
from VuVoPy.data.utils.fundamental_frequency import FundamentalFrequency as ff

def hnr(folder_path, plim=(30, 500), hop_size = 512, dlog2p=1/96, dERBs=0.1, sTHR=-np.inf):
    """
    Compute the Harmonics-to-Noise Ratio (HNR) of an audio signal.

    HNR is a ratio of periodic (harmonic) to aperiodic (noise) energy in a signal,
    widely used in speech signal analysis to assess voice quality.

    Args:
        folder_path (str): Path to the audio file in WAV format.
        plim (tuple, optional): Pitch range limits in Hz. Default is (30, 500).
        hop_size (int, optional): Hop size for analysis. Default is 512.
        dlog2p (float, optional): Logarithmic pitch step size. Default is 1/96.
        dERBs (float, optional): Step size in ERB scale. Default is 0.1.
        sTHR (float, optional): Silence threshold in dB. Default is -np.inf.

    Returns:
        float: Mean HNR value. Returns NaN if no valid values are found.

    Notes:
        - Signal is preprocessed and pitch-tracked before HNR estimation.
        - Frames with invalid or low-frequency F0 are excluded.
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
