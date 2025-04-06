import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import scipy.signal
from scipy.stats import mode
from data.containers.prepocessing import Preprocessed as pp
from data.containers.sample import VoiceSample as vs
from data.containers.segmentation import Segmented as sg

class Vuvs:
    """Class to detect voiced/unvoiced/scilence segments in an audio signal using GMM."""
    
    def __init__(self, segment, fs, winlen = 512, winover = 256, wintype = 'hann', smoothing_window=5):
        self.segment = segment.get_segment()
        self.segment_prem = segment.get_preem_segment()
        self.segment_norm = segment.get_norm_segment()
        self.fs = fs
        self.winlen = winlen
        self.winover = winover
        self.wintype = wintype
        self.smoothing_window = smoothing_window

        # Compute vuvs vuvs upon initialization
        self.vuvs = self.calculate_vuvs()
        
    def calculate_vuvs(self):
        """Compute voiced/unvoiced/scilence segments using GMM."""
       #TODO: Implement the GMM algorithm to classify segments as voiced, unvoiced, or silence. 
        return  vuvs_gmm(self.segment, self.fs, self.smoothing_window)
    def get_sampling_rate(self):
        """Return the sampling rate."""
        return self.fs