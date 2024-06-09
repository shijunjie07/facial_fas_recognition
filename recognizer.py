# --------------------------
# face recognition implentation
# @author: Shi Junjie
# # Sat 3 June 2024
# --------------------------

import torch
from facenet_pytorch import InceptionResnetV1

class Recognizer:
    
    def __init__(self):
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
            
    def encode(self, img):
        res = self.resnet(torch.Tensor(img))
        return res