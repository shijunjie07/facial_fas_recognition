# --------------------------
# face recognition implentation
# @author: Shi Junjie
# # Sat 3 June 2024
# --------------------------

import torch
from .config import Config
from facenet_pytorch import InceptionResnetV1

class Recognizer:
    
    def __init__(self):
        # self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        self.resnet = InceptionResnetV1(pretrained=None)
        self.resnet.load_state_dict(
            torch.load(Config.RECOGNIZER_WEIGHTS, map_location=torch.device('cpu'))['state_dict']
        )
        self.resnet.eval()
            
    def encode(self, img):
        res = self.resnet(torch.Tensor(img))
        return res