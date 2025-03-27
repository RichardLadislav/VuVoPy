import numpy as np
import librosa as lb
from containers.segmentation import Segmentred


class FormantFrequencies(Segmentred):
    '''Class to compute F1 and F2 frormant frequencies'''
    def __init__(self, x, fs, xnorm, preem, xsegment, winlen, wintype, winover, formants, alpha=0.94):
        super().__init__(self, x, fs, xnorm, preem, xsegment, winlen, wintype, winover, alpha=0.94)
        self.formants = formants
    @classmethod
    def from_voice_sample(cls, segments):
        seg_x = segments.get_segment() 
        seg_x_preem = segments.get_preem_segment()
        seg_x_norm = segments.get_norm_segment()
        fs = segments.get_sampling_rate() 
        
        order = np.fix(fs/1000 +2)

        lpc_coeff_x = lb.lpc(seg_x, order=order)
        lpc_coeff_x_prem = lb.lpc(seg_x_preem ,order=order)
        lpc_coeff_x = lb.lpc(seg_x_norm, order=order)
        

        

        
        

