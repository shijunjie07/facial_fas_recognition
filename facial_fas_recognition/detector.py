# --------------------------
# face detection implentation
# @author: Shi Junjie
# Sat 3 June 2024
# --------------------------

import torch
from .config import Config
from facenet_pytorch import MTCNN


class FaceDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.mtcnn = MTCNN(
            image_size=224, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60,
        )
        self.mtcnn.load_state_dict(
            torch.load(Config().DETECTOR_WEIGHTS, map_location=self.device)
        )
        self.mtcnn.eval()
        
    def detect_faces(self, image):
        boxes, _ = self.mtcnn.detect(image)
        cropped_faces = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cropped_faces.append(image[y1:y2, x1:x2])
        return boxes, cropped_faces