import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import mode
import time 
import librosa

def vuvs_gmm(segments, sr, winover, smoothing_window=5):
    """
    Classify segments of an audio signal as voiced, unvoiced, or silence using Gaussian Mixture Models (GMM).
    
    Parameters:
    - y : np.ndarray : Audio signal.
    - sr : int : Sampling rate.
    - smoothing_window : int : Size of the smoothing window for the GMM classification.
    
    Returns:
    - np.ndarray : Array of classified segments (0: unvoiced, 1: voiced, 2: silence).
    """

    features = []
    frame_length = segments.shape[1]
    segments = segments.T  # Transpose to iterate over frames

    for frame in segments:
        spectrum = np.abs(np.fft.rfft(frame, n=frame_length))
        freqs = np.fft.rfftfreq(frame_length, 1 / sr)

        # E: Frame energy above 200 Hz
        mask = freqs > 200
        E = 10 * np.log10(np.sum(spectrum[mask] ** 2) + 1e-10)

        # Ehi: High-frequency to low-frequency ratio
        mid = int(len(freqs) * 0.25)
        low_energy = np.sum(spectrum[:mid] ** 2)
        high_energy = np.sum(spectrum[mid:] ** 2)
        Ehi = 10 * np.log10(high_energy / (low_energy + 1e-10) + 1e-10)

        # C1: Normalized autocorrelation coefficient
        s_prev = frame[:-1] #if count > 0 else frame
        C1 = np.correlate(frame, s_prev[:frame_length])[0] / (np.sum(frame ** 2) + 1e-10)
        # Nz: Zero-crossing rate 
 
        zcr =np.sum(np.diff(np.sign(frame)) != 0)
        features.append([E, 100 * C1, Ehi, zcr])
    features = np.array(features)

    # Classify voiced/unvoiced/silence using GMM
    gmm1 = GaussianMixture(n_components=2, covariance_type='diag', random_state=0, max_iter=100)
    gmm1.fit(features)  
    means = gmm1.means_
    voiced_idx = np.argmax(means[:, 0])  # Higher energy => Voiced
    voiced_mask = gmm1.predict(features) == voiced_idx

    # Unvoiced vs Silence
    features_us = features[~voiced_mask]
    gmm2 = GaussianMixture(n_components=2, covariance_type='diag', random_state=0, max_iter=100)
    gmm2.fit(features_us)
    means2 = gmm2.means_
    unvoiced_idx = np.argmax(means2[:, 0])
    unvoiced_mask = gmm2.predict(features_us) == unvoiced_idx
    
    # Raw labels: 0 = silence, 1 = unvoiced, 2 = voiced
    labels = np.zeros(len(features), dtype=int)
    labels[voiced_mask] = 2  # Voiced
    labels[~voiced_mask] = np.where(unvoiced_mask, 1, 0)  # Unvoiced
    
    #Step 1: Smoothing the labels
    half_win = smoothing_window // 2
    padded = np.pad(labels, (half_win, half_win), mode='edge')
    
    smoothed_labels = np.array([
        int(mode(padded[i:i + smoothing_window], keepdims=False).mode)
        for i in range(len(labels))
    ])

    # Step 2: Post-processing cleanup
    frame_duration = (frame_length-winover) / sr  # seconds
    min_duration_frames = int(0.01 / frame_duration)  #  10ms
    long_wait_frames = int(0.05 / frame_duration)  # 90ms
    
    labels = smoothed_labels.copy()
    def relabel_short_segments(target_class, surrounding_class):
        i = 0
        while i < len(labels):
            if labels[i] != target_class:
                i += 1
                continue
            start = i
            while i < len(labels) and labels[i] == target_class:
                i += 1
            length = i - start
            if length < min_duration_frames:
                before = labels[start - 1] if start > 0 else -1
                after = labels[i] if i < len(labels) else -1
                if before == after == surrounding_class:
                    labels[start:i] = surrounding_class

    # Rule A: Short voiced segments between unvoiced → unvoiced
    relabel_short_segments(target_class=2, surrounding_class=1)

    # Rule B: Short unvoiced segments between voiced → voiced
    relabel_short_segments(target_class=1, surrounding_class=2)

    # Rule C: First unvoiced segment not followed by voiced in 90ms → silence
    # Find first UV segment
    for i in range(len(labels)):
        if labels[i] == 1:
            lookahead = labels[i:i + long_wait_frames]
            if 2 not in lookahead:
                labels[i:i + len(lookahead)] = 0  # Silence
            break
    
    return labels