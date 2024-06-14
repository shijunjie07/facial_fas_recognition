# --------------------------
# config
# @author: Shi Junjie
# Tue 11 June 2024
# --------------------------

import os

class Config:
    
    def __init__(self):
        
        self.pretriained_weights_dir = '/home/junja/facial_fas_recognition/pretrained'
        
        # set weights file path
        self.DETECTOR_WEIGHTS = os.path.join(
            self.pretriained_weights_dir, 'detector.pt')
        self.SPOOF_WEIGHTS = os.path.join(
            self.pretriained_weights_dir, 'spoof.pt')
        self.RECOGNIZER_WEIGHTS = os.path.join(
            self.pretriained_weights_dir, 'recognizer.pt')

    def set_directory(self, directory):
        
        self.pretriained_weights_dir = directory
        
        self.DETECTOR_WEIGHTS = os.path.join(
            self.pretriained_weights_dir, 'detector.pt')
        self.SPOOF_WEIGHTS = os.path.join(
            self.pretriained_weights_dir, 'spoof.pt')
        self.RECOGNIZER_WEIGHTS = os.path.join(
            self.pretriained_weights_dir, 'recognizer.pt')


    