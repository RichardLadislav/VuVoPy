import numpy as np
from sample import VoiceSample
from prepocessing import Preprocessed

class Segemented(Preprocessed):
    def __init__(self, x, fs, xnorm, preem, xsegment, winlen, wintype, winover, alpha=0.94):
        super().__init__(x, fs, xnorm, preem, alpha)
        self.xsegment = xsegment if xsegment is not None else x
        self.winlen = winlen
        self.wintype = wintype
        self.winover = winover