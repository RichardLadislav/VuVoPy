import numpy as np
from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.containers.segmentation import Segmented as sg

def hnr(folder_path, winlen=512, winover=256 , wintype='hann', f0_min=75, f0_max=500):
    """
    Compute Harmonics-to-Noise Ratio (HNR) using an autocorrelation-based method.

    This function processes a WAV file, divides it into overlapping frames, and estimates
    the harmonic-to-noise ratio (HNR) for each frame using pitch period information
    derived from the autocorrelation method.

    Args:
        folder_path (str): Path to the audio file (WAV format).
        winlen (int): Frame length in samples.
        winover (int): Overlap between consecutive frames in samples.
        wintype (str): Type of window function to apply (e.g., 'hann', 'hamming').
        f0_min (float): Minimum fundamental frequency in Hz.
        f0_max (float): Maximum fundamental frequency in Hz.

    Returns:
        float: Mean HNR value across all frames.
    """

    # Load and preprocess the audio file
    segment = sg.from_voice_sample(pp.from_voice_sample(vs.from_wav(folder_path)), winlen, wintype, winover)
    signal = segment.get_norm_segment().T  # Transpose to get shape (num_frames, num_samples)
    fs = segment.get_sampling_rate()  # Get sampling rate

    hnr_values = []
    num_frames = signal.shape[0]

    for i in range(num_frames):
        frame = signal[i, :]
        frame = frame - np.mean(frame)  # Remove DC offset
        frame = frame / (np.max(np.abs(frame)) + 1e-10)  # Normalize to prevent floating-point errors

        if np.max(np.abs(frame)) < 1e-6:
            continue  # Skip silent frames

        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]  # Keep only positive lags
        autocorr /= np.max(autocorr)  # Normalize to 1

        # Find fundamental period (within F0 range)
        min_period = int(fs / f0_max)
        max_period = min(int(fs / f0_min), len(autocorr) - 1)

        if max_period <= min_period:
            continue  # Skip unreliable frames

        # Find the actual F0 peak (ignore zero lag)
        peak_idx = np.argmax(autocorr[min_period:max_period]) + min_period
        r_max = autocorr[peak_idx]  # Peak value at estimated fundamental period

        # Ensure r_max is in valid range
        r_max = np.clip(r_max, 1e-4, 0.999)  # Avoid log(0) issues

        # Compute HNR
        hnr = 10 * np.log10(r_max / (1 - r_max))
        hnr_values.append(hnr)

    return np.mean(hnr_values) if hnr_values else float('nan')
