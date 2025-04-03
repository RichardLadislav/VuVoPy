import numpy as np
from scipy.signal import find_peaks
from data.containers.prepocessing import Preprocessed as pp
from data.containers.sample import VoiceSample as vs
from data.containers.segmentation import Segmented as sg
from data.utils.fundamental_frequency import FundamentalFrequency as ff

def hnr_fft(folder_path, winlen=1608, winover=804, wintype='hann', f0_min=75, f0_max=500):
    """
    Compute HNR using an FFT-based spectral method.

    Parameters:
    - folder_path: Path to the audio file  
    - winlen: Frame length in frames
    - winover: Overlap in frames
    - wintype: Window type
    - f0_min: Minimum fundamental frequency (Hz)
    - f0_max: Maximum fundamental frequency (Hz)

    Returns:
    - Mean HNR value across frames (in dB)
    """

    # Load and preprocess the audio file
    segment = sg.from_voice_sample(pp.from_voice_sample(vs.from_wav(folder_path)), winlen, wintype, winover)
    signal = segment.get_norm_segment().T  # Transpose to (num_frames, num_samples)
    fs = segment.get_sampling_rate()

    hnr_values = []
    epsilon = 1e-10  # To avoid log(0)

    for frame in signal:
        # Apply FFT
        spectrum = np.abs(np.fft.rfft(frame)) ** 2  # Power spectrum
        freqs = np.fft.rfftfreq(len(frame), 1 / fs)

        # Estimate F0 using FundamentalFrequency class
        f0 = ff.get_f0(frame)
        if np.isnan(f0):
            continue  # Skip frame if F0 is not detected

        # Find harmonics (multiples of F0 within the spectrum)
        harmonic_indices = [np.argmin(np.abs(freqs - (n * f0))) for n in range(1, 6) if (n * f0) < (fs / 2)]

        # Compute power of harmonics
        harmonic_power = np.sum(spectrum[harmonic_indices])

        # Compute noise power (total power minus harmonic power)
        noise_power = np.sum(spectrum) - harmonic_power

        # Compute HNR in dB
        if noise_power > epsilon:
            hnr = 10 * np.log10(harmonic_power / noise_power)
            hnr_values.append(hnr)

    return np.mean(hnr_values) if hnr_values else float('nan')
if __name__ == "__main__":
    file_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//vowel_e_test.wav"
    # Compute HNR
    hnr_value = hnr_fft(file_path)
    if np.isnan(hnr_value):
        print("Could not compute HNR.")
    else:
        print(f"Mean HNR: {hnr_value:.2f} dB")
