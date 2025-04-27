import numpy as np
from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.containers.segmentation import Segmented as sg

def hnr(folder_path, winlen=512, winover=256 , wintype='hann', f0_min=75, f0_max=500):
    """
    Compute Harmonics-to-Noise Ratio (HNR) using an autocorrelation-based method.

    Parameters:
    - folder_path: Path to the audio file  
    - winlen: Frame length in samples
    - winover: Overlap in samples
    - wintype: Window type
    - f0_min: Minimum fundamental frequency (Hz)
    - f0_max: Maximum fundamental frequency (Hz)

    Returns:
    - Mean HNR value across frames
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

if __name__ == "__main__":
    file_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//recordings-20250306T192251Z-001//recordings//K1020//K1020_7.1-1-a_1.wav"
    
    # Compute HNR
    hnr_value = hnr(file_path)
    if np.isnan(hnr_value):
        print("Could not compute HNR.")
    else:
        print(f"Mean HNR: {hnr_value:.2f} dB")
