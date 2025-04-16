import numpy as np  
from data.containers.prepocessing import Preprocessed as pp
from data.containers.sample import VoiceSample as vs
from data.utils.fundamental_frequency import FundamentalFrequency as f0
from data.containers.segmentation import Segmented as sg
from data.utils.vuvs_detection import Vuvs as vuvs
from data.containers.voiced_sample import VoicedSample as vos

def shimmerAPQ(folder_path, n_points=5, plim=(30, 500), sTHR=0.5):
    """
    Calculate shimmer APQ-N: amplitude perturbation quotient over N-point window.

    Args:
        folder_path (str): Path to .wav file.
        n_points (int): Number of points in local averaging window (e.g., 3 for APQ3, 5 for APQ5).
        plim (tuple): F0 pitch range (Hz).
        sTHR (float): Voicing threshold for F0 tracking.

    Returns:
        float: shimmer APQ-N value
    """
    sample = vs.from_wav(folder_path)
    signal = sample.get_waveform()
    sr = sample.get_sampling_rate()

    # Extract F0 and keep only voiced
    f0_track = f0(sample, plim=plim, sTHR=sTHR).get_f0()
    f0_track = f0_track[f0_track > 30]
    if len(f0_track) < n_points:
        return 0

    # Approximate cycle boundaries using median F0
    cycle_len = int(sr / np.median(f0_track))
    starts = np.arange(0, len(signal) - cycle_len, cycle_len)

    amplitudes = []
    for start in starts:
        cycle = signal[start:start + cycle_len]
        if len(cycle) < cycle_len:
            continue
        amp = np.max(cycle) - np.min(cycle)
        amplitudes.append(amp)

    amplitudes = np.array(amplitudes)
    if len(amplitudes) < n_points:
        return 0

    global_mean_amp = np.mean(amplitudes)
    if global_mean_amp == 0:
        return 0

    # Smoothed local average using moving average (centered)
    kernel = np.ones(n_points) / n_points
    smoothed = np.convolve(amplitudes, kernel, mode='valid')

    # Align original amplitude vector to the smoothed one
    offset = n_points // 2
    trimmed = amplitudes[offset:len(amplitudes) - offset]

    if len(trimmed) != len(smoothed):
        return 0

    shimmer_vals = np.abs(trimmed - smoothed) / global_mean_amp
    return np.mean(shimmer_vals) if len(shimmer_vals) else 0

def main():
    #folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//activity_unproductive.wav"
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//PARCZ_complet//recordings//K1021//K1021_8.2-1_1.wav"
    out = shimmerAPQ(folder_path, n_points=3, sTHR=0.5)
    print(out)
if __name__ == "__main__": 
    main()