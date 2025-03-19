import librosa


class VoiceSample(object):
    """Creating sample class from audio"""

    def __init__(self, x, fs):
        """Constructor method"""
        self.x = x 
        self.fs = fs
        
        
    @classmethod
    def from_wav(file_path, sr=None):
        "Initializes VoiceSample object from wav file"
        return librosa.load(file_path, sr) 

