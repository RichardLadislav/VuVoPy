import numpy as np
import librosa as lb
import matplotlib.pyplot as plt
from data.containers.prepocessing import Preprocessed     
from data.containers.sample import VoiceSample
from data.containers.segmentation import Segmented


class FormantFrequencies(Segmented):
    '''Class to compute F1 and F2 frormant frequencies'''
    def __init__(self, x, fs, xnorm, preem, xsegment, winlen, wintype, winover, formants, alpha=0.94):
        super().__init__(x, fs, xnorm, preem, xsegment, winlen, wintype, winover, alpha=0.94)
        self.formants = formants
    @classmethod
    def from_voice_sample(cls, segments):
        """
        Creates an instance of the class from a voice sample by extracting formant frequencies.
        Args:
            cls: The class itself, used to create an instance.
            segments: An object containing segmented voice data with methods to retrieve
                      raw, pre-emphasized, and normalized segments, as well as the sampling rate.
        Returns:
            An instance of the class initialized with the sampling rate and extracted formant frequencies.
        Notes:
            - The method calculates LPC coefficients for raw, pre-emphasized, and normalized segments.
            - Formant frequencies are derived from the roots of the LPC polynomial.
            - Only roots with non-negative imaginary parts are considered.
            - The method currently extracts and sorts the first three formants for each segment.
            - The bandwidths of the formants are not calculated at this stage.
        """
        seg_x = segments.get_segment() 
        seg_x_preem = segments.get_preem_segment()
        seg_x_norm = segments.get_norm_segment()
        fs = segments.get_sampling_rate() 
        
        order = int(np.fix(fs/1000 +2))

        lpc_coeff_x = lb.lpc(seg_x, order=order)
        lpc_coeff_x_prem = lb.lpc(seg_x_preem ,order=order)
        lpc_coeff_x_norm = lb.lpc(seg_x_norm, order=order)
        
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
            #TODO: Bandwidths of formants are not calculated yet
            
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

def main():
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//vowel_e_test.wav"
    processed_sample = Preprocessed.from_voice_sample(VoiceSample.from_wav(folder_path))

    seg = Segmented.from_voice_sample(processed_sample, winlen=512, winover=256, wintype="hann")
    formants = FormantFrequencies.from_voice_sample(seg)
    formants_x = formants.get_formants()
    formants_preem = formants.get_formants_preem()
    formants_xnormm = formants.get_formants_norm()
    # Apply different windowing functions dynamically
    #seg_wave = seg.get_segment(winlen=512, wintype="hann", winover=256)
    #seg_preem = seg.get_preem_segment(winlen=512, wintype="blackman", winover=256)
    #seg_norm = seg.get_norm_segment(winlen=512, wintype="square", winover=256)

    plt.figure(figsize=(10, 6))
    for i in range(formants_preem.shape[1]):  # Iterate over formant frequency columns
        plt.scatter( formants_preem[:, i], label=f'Formant {i}',marker="x")

    plt.title("Formant Frequencies Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid()
    plt.show(block=True) 
    print("holap")
if __name__ == "__main__": 
    main()


            

        
        
        

