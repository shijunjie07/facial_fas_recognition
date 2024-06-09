# --------------------------
# face detection implentation
# @author: Shi Junjie
# Sat 3 June 2024
# --------------------------

from facenet_pytorch import MTCNN


class FaceDetector:
    def __init__(self):
        self.mtcnn = MTCNN(
            image_size=224, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60
        )
        # self.detect_box = MethodType(self._detect_faces, self.mtcnn)
        
    def detect_faces(self, image):
        boxes, _ = self.mtcnn.detect(image)
        cropped_faces = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cropped_faces.append(image[y1:y2, x1:x2])
        return boxes, cropped_faces