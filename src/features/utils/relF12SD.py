import numpy as np  
from data.containers.prepocessing import Preprocessed as pp
from data.containers.sample import VoiceSample as vs
from data.containers.segmentation import Segmented as sg
from data.utils.formant_frequencies import FormantFrequencies as ff

def relF1SD(folder_path, winlen = 512, winover = 256, wintype = 'hann'):

    formant_freqs = ff.from_voice_sample(sg.from_voice_sample(pp.from_voice_sample(vs.from_wav(folder_path)),winlen, wintype ,winover))
    return np.mean(formant_freqs.get_formants_preem()[:,0])/np.std(formant_freqs.get_formants_preem()[:,0])
#def relF1SD(segments, fs, formants):

def relF2SD(folder_path, winlen = 512, winover = 256, wintype = 'hann'):

    formant_freqs = ff.from_voice_sample(sg.from_voice_sample(pp.from_voice_sample(vs.from_wav(folder_path)),winlen, wintype ,winover))
    return np.mean(formant_freqs.get_formants_preem()[:,1])/np.std(formant_freqs.get_formants_preem()[:,1])

def main():
    folder_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//vowel_e_test.wav"
    out = relF1SD(folder_path, wintype="hamm")
    out1 = relF2SD(folder_path)
    print(out)
    print(out1)
if __name__ == "__main__": 
    main()