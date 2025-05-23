import numpy as np  
from VuVoPy.data.containers.prepocessing import Preprocessed as pp
from VuVoPy.data.containers.sample import VoiceSample as vs
from VuVoPy.data.utils.fundamental_frequency import FundamentalFrequency as f0
from VuVoPy.data.containers.segmentation import Segmented as sg
from VuVoPy.data.utils.vuvs_detection import Vuvs as vuvs
from VuVoPy.data.containers.voiced_sample import VoicedSample as vos

def shimmerAPQ(folder_path, n_points=5, plim=(30, 500), sTHR=0.5, winlen = 512, winover = 496 , wintype = 'hamm'):
    """
    Calculate shimmer APQ-N: amplitude perturbation quotient over an N-point window.

    This function estimates shimmer by analyzing cycle-to-cycle amplitude variations 
    in voiced frames of an audio signal. The average absolute difference between local 
    peak amplitudes is computed over a moving window of size `n_points`.

    Args:
        folder_path (str): Path to the .wav file.
        n_points (int): Number of points in the local averaging window (e.g., 3 for APQ3, 5 for APQ5).
        plim (tuple): F0 pitch range in Hz. Default is (30, 500).
        sTHR (float): Voicing threshold for F0 tracking.
        winlen (int): Window length for segmentation.
        winover (int): Window overlap for segmentation.
        wintype (str): Window type for segmentation.

    Returns:
        float: Shimmer APQ-N value.
    """
    preprocessed_sample = pp.from_voice_sample(vs.from_wav(folder_path))
    segment = sg.from_voice_sample(preprocessed_sample, winlen, wintype, winover)
    fs = segment.get_sampling_rate()
    labels = vuvs(segment, fs=fs, winlen =segment.get_window_length(), winover = segment.get_window_overlap(), wintype=segment.get_window_type(), smoothing_window=5)
    silence_removed_sample = vos(preprocessed_sample, labels, fs)
    signal = silence_removed_sample.get_silence_remove_sample()
    sr = silence_removed_sample.get_sampling_rate()

    # Extract F0 and keep only voiced
    f0_track = f0(silence_removed_sample, plim=plim, sTHR=sTHR).get_f0()
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
