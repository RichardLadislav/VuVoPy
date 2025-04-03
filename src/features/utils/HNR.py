import numpy as np
from data.containers.prepocessing import Preprocessed as pp
from data.containers.sample import VoiceSample as vs
from data.containers.segmentation import Segmented as sg
from data.utils.fundamental_frequency import FundamentalFrequency as ff

def hnr(folder_path, winlen = 1600, winover = 800, wintype = 'hann', f0_min = 75, f0_max = 500):
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
    segment = sg.from_voice_sample(pp.from_voice_sample(vs.from_wav(folder_path)), winlen, wintype, winover)
    signal = segment.get_norm_segment().T  # Transpose to get shape (num_samples, num_frames)   
    fs = segment.get_sampling_rate()  # Get sampling rate

    # Compute fundamental frequency using FundamentalFrequency class

    hnr_values = []
    num_frames = signal.shape[0] 
    for i in range(num_frames):
        autocorr = np.correlate(signal[i,:], signal[i,:], mode='full') / np.dot(signal[i,:], signal[i,:])
        autocorr = autocorr[len(autocorr) // 2:]  # Keep only positive lags
        autocorr /= np.max(np.abs(autocorr))

        # Find fundamental period (max autocorr peak within f0 range)
        min_period = int(fs / f0_max)  # Corresponds to max frequency
        max_period = int(fs / f0_min)  # Corresponds to min frequency

        if max_period >= len(autocorr):
            max_period = len(autocorr) - 1

        peak_idx = np.argmax(autocorr[min_period:max_period]) + min_period
        r_max = autocorr[peak_idx]  # Max autocorrelation peak

        # Compute HNR using the definition
        if r_max > 0.99:
            r_max = 0.99  # Prevent extreme values
        hnr = 10 * np.log10((r_max + 1e-6) / (1 - r_max + 1e-6))
        hnr_values.append(hnr)

    if len(hnr_values) > 0:
        return np.mean(hnr_values)
    else:
        return float('nan')

if __name__ == "__main__":
    file_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//sine_wave_with_noise.wav"
    # Compute HNe
    hnr_value = hnr(file_path)
    if np.isnan(hnr_value):
        print("Could not compute HNR.")
    else:
        print(f"Mean HNR: {hnr_value:.2f} dB")
