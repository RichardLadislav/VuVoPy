import numpy as np
from scipy.signal import get_window
from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.containers.segmentation import Segmented as sg
from VuVoPy.data.utils.fundamental_frequency import FundamentalFrequency as ff

def hnr_fft(folder_path, winlen=1608, winover=804, wintype='hann', f0_min=75, f0_max=500):
    """
    Compute HNR using an FFT-based spectral method.

    Parameters:
    - folder_path: Path to the audio file  
    - winlen: Frame length in samples
    - winover: Overlap in samples
    - wintype: Window type
    - f0_min: Minimum fundamental frequency (Hz)
    - f0_max: Maximum fundamental frequency (Hz)

    Returns:
    - Mean HNR value across frames (in dB)
    """

    # Load and preprocess the audio file
    voice_sample = vs.from_wav(folder_path)
    processed_sample = pp.from_voice_sample(voice_sample)
    signal = processed_sample.get_preemphasis()
    fs = voice_sample.get_sampling_rate()  # Get sampling rate

    # Compute fundamental frequency using FundamentalFrequency class
    f0_array = ff(voice_sample, plim=(f0_min, f0_max), hop_size=winover, dlog2p=1/96, dERBs=0.1, sTHR=-np.inf).get_f0()

    # Ensure F0 array matches frame count
    num_frames = (len(signal) - winlen) // winover + 1
    f0_array = np.pad(f0_array, (0, max(0, num_frames - len(f0_array))), mode='edge')

    hnr_values = []
    epsilon = 1e-10  # To avoid log(0)

    # Define window function
    window = get_window(wintype, winlen)

    # Process each frame
    for i in range(num_frames):
        start = i * winover
        end = start + winlen
        if end > len(signal):
            break
        
        frame = signal[start:end] * window  # Apply window
        spectrum = np.abs(np.fft.rfft(frame)) ** 2  # Power spectrum
        freqs = np.fft.rfftfreq(winlen, 1 / fs)

        # Get F0 for this frame
        f0 = f0_array[i]
        if np.isnan(f0) or f0 < f0_min or f0 > f0_max:
            continue  # Skip unreliable F0 values

        # Find harmonic indices in FFT spectrum
        harmonic_indices = [np.argmin(np.abs(freqs - (n * f0))) for n in range(1, 6) if (n * f0) < (fs / 2)]

        # Compute harmonic and noise power
        harmonic_power = np.sum(spectrum[harmonic_indices]) if harmonic_indices else 0
        total_power = np.sum(spectrum)
        noise_power = max(total_power - harmonic_power, epsilon)  # Ensure non-zero noise power

        # Compute HNR in dB
        if harmonic_power > 0:
            hnr = 10 * np.log10(harmonic_power / noise_power)
            hnr_values.append(hnr)

    return np.mean(hnr_values) if hnr_values else float('nan')
