# config

import os

class Config:
    curr_dir = os.getcwd()
    # pretrained weights
    DETECTOR_WEIGHTS = os.path.join(curr_dir, 'pretrained/detector.pt')
    SPOOF_WEIGHTS = os.path.join(curr_dir, 'pretrained/spoof.pt')
    RECOGNIZER_WEIGHTS = os.path.join(curr_dir, 'pretrained/recognizer.pt')
    
    
# if __name__ == "__main__":
#     print(Config.curr_dir)
#     print(Config.DETECTOR_WEIGHTS)
#     print(Config.SPOOF_WEIGHTS)
#     print(Config.RECOGNIZER_WEIGHTS)
    