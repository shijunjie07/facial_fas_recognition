# --------------------------
# face recognition implentation
# @author: Shi Junjie
# Sat 3 June 2024
# --------------------------

import torch
from .config import Config
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

class Recognizer:
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.resnet = InceptionResnetV1(pretrained=None)
        self.resnet.load_state_dict(
            torch.load(Config().RECOGNIZER_WEIGHTS, map_location=self.device),
            strict=False
        )
        self.resnet.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def encode(self, img):
        with torch.no_grad():
            res = self.resnet(self.transform(img).unsqueeze(0))
        return res