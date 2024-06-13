# --------------------------
# config
# @author: Shi Junjie
# Tue 11 June 2024
# --------------------------

import os

class Config:
    
    # current dir
    curr_dir = os.getcwd()
    
    # pretrained weights
    DETECTOR_WEIGHTS = os.path.join(curr_dir, 'pretrained/detector.pt')
    SPOOF_WEIGHTS = os.path.join(curr_dir, 'pretrained/spoof.pt')
    RECOGNIZER_WEIGHTS = os.path.join(curr_dir, 'pretrained/recognizer.pt')
    
    # Known Face dir
    KNOWN_FACE_DIR = os.path.join(curr_dir, 'faces')