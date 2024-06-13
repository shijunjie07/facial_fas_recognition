# --------------------------
# face anti-spoofing implentation
# @author: Shi Junjie
# Sat 3 June 2024
# --------------------------


import torch
from torch import nn
from .config import Config
from torchvision import transforms
from torchvision.models import mobilenet_v2


class SpoofNet(nn.Module):
    
    def __init__(self):
        super(SpoofNet, self).__init__()
        
        # pretrained MobileNetV2
        self.pretrained_net = mobilenet_v2(pretrained=True)
        self.features = self.pretrained_net.features
        
        # extra layers
        self.conv2d = nn.Conv2d(1280, 32, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.conv2d(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
class AntiSpoof:
    
    def __init__(self):
        self.model = SpoofNet()
        self.model.load_state_dict(
            torch.load(Config.SPOOF_WEIGHTS, map_location=torch.device('cpu'))['state_dict'])
        self.device = 'cpu'
        self.model.to(self.device)
        
        # evaluation
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        width, height = image.size
        
        if width == 0 or height == 0:
            return False, False, 0.0
        
        norm_features = self.transform(image).unsqueeze(0)
        norm_features = norm_features.to(self.device)
        with torch.no_grad():
            output = self.model(norm_features)
        predicted = (output > 0.5).bool()
        
        return True, predicted, output[0, 0]