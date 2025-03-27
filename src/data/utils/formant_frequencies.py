import numpy as np
import librosa as lb
from containers.segmentation import Segmentred


class FormantFrequencies(Segmentred):
    '''Class to compute F1 and F2 frormant frequencies'''
    def __init__(self, x, fs, xnorm, preem, xsegment, winlen, wintype, winover, formants, alpha=0.94):
        super().__init__(x, fs, xnorm, preem, xsegment, winlen, wintype, winover, alpha=0.94)
        self.formants = formants
    @classmethod
    def from_voice_sample(cls, segments):
        seg_x = segments.get_segment() 
        seg_x_preem = segments.get_preem_segment()
        seg_x_norm = segments.get_norm_segment()
        fs = segments.get_sampling_rate() 
        
        order = int(np.fix(fs/1000 +2))

        lpc_coeff_x = lb.lpc(seg_x, order)
        lpc_coeff_x_prem = lb.lpc(seg_x_preem ,order)
        lpc_coeff_x_norm = lb.lpc(seg_x_norm, order)
        
        N = lpc_coeff_x.shape[0]

        formants = np.zeros((N,3,3))
        #TODO: Zatial len formantu, sirka pasem mozno potom,
        
        for i in range(N):
            #Findiung roots of nominator of transfer function
            rts_x = np.roots(lpc_coeff_x)
            rts_x_preem = np.roots(lpc_coeff_x_prem)
            rts_x_norm = np.roots(lpc_coeff_x_norm)
            
            #Finding non-zero Im{Z} >=0
            rts_x = rts_x[(np.imag(rts_x)>=0 )].copy()
            rts_x_preem = rts_x_preem[(np.imag(rts_x)>=0 )].copy()
            rts_x_norm = rts_x_norm[(np.imag(rts_x)>=0 )].copy()

            #Finding formants
            tempF_x = np.arctan2(np.imag(rts_x),np.real(rts_x))
            tempF_x_preem = np.arctan2(np.imag(rts_x_preem),np.real(rts_x_preem))
            tempF_x_norm = np.arctan2(np.imag(rts_x_norm),np.real(rts_x_norm))

            #Sorting formants
            sort_F = sorted(tempF_x)
            sort_F_preem = sorted(tempF_x_preem)
            sort_F_norm = sorted(tempF_x_preem)
            
            formants[i,:,0] =  sort_F[0,2]
            formants[i,:,1] =  sort_F_preem[0,2]
            formants[i,:,2] =  sort_F_norm[0,2]
            
            return cls(fs, formants)
        
    def get_formants(self):
        """Return the numpy array of formants extracted from raw waveform"""
        return self.formants[:,:,0]

    def get_formants_preem(self):
        """Return the numpy array of formants extracted from pre-emphasis waveform"""
        return self.formants[:,:,1]
    
    
    def get_formants_norm(self):
        """Return the numpy array of formants extracted from normalized waveform"""
        return self.formants[:,:,2]
    
    def get_sampling_rate(self):
        """Return the sampling rate."""
        return self.fs



            

        
        
        

