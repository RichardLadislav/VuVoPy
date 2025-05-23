import numpy as np
from scipy.signal import find_peaks
from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.containers.segmentation import Segmented as sg
from VuVoPy.data.utils.fundamental_frequency import FundamentalFrequency as ff

def hnr_cepstrum(folder_path, winlen=1024, winover=512, wintype='hann', f0_min=75, f0_max=500):
    """
    Compute HNR using cepstral analysis (quefrency method).

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
        # Compute the log power spectrum
        spectrum = np.abs(np.fft.fft(frame)) ** 2
        log_spectrum = np.log(spectrum + epsilon)

        # Compute cepstrum (inverse FFT of log spectrum)
        cepstrum = np.fft.ifft(log_spectrum).real

        # Define quefrency range corresponding to valid F0
        min_quefrency = int(fs / f0_max)
        max_quefrency = int(fs / f0_min)

        if min_quefrency >= max_quefrency:
            continue  # Skip frame if range is invalid

        # Find peak in valid quefrency range
        peak_idx = np.argmax(cepstrum[min_quefrency:max_quefrency]) + min_quefrency
        r_max = cepstrum[peak_idx]  # Maximum cepstral peak

        # Compute HNR relative to noise floor
        noise_floor = np.mean(cepstrum[min_quefrency:max_quefrency])  # Average background noise
        if noise_floor > epsilon:
            hnr = 10 * np.log10(r_max / noise_floor)
            hnr_values.append(hnr)

    return np.mean(hnr_values) if hnr_values else float('nan')
