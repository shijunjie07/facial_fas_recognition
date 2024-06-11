# --------------------------
# main
# @author: Shi Junjie
# # Sun 9 June 2024
# --------------------------

import numpy as np
from PIL import Image

# local
from detector import FaceDetector
from fas import AntiSpoof
from recognizer import Recognizer


class FaceRecg:
    
    def __init__(self, embeddings:dict, recg_thresh=1.2):
        self.embeddings = embeddings
        self.recg_thresh = recg_thresh
        self.face_detector = FaceDetector()
        self.liveness_checker = AntiSpoof()
        self.recognizer = Recognizer()
        
    def recg(self, image:np.array) -> tuple[bool, dict]:
        recg_faces = {
            "faces": [],
            "face_ids": [],
            "livenesses": [],
            "liveness_scores": [],
            "bboxes": [],
        }
        
        # detect face
        batch_boxes, cropped_faces = self.face_detector.detect_faces(image)
        
        if cropped_faces is not None and batch_boxes is not None:
            for box, face in zip(batch_boxes, cropped_faces):
                face_pil = Image.fromarray(face.transpose(2, 0, 1))
                # check liveness
                valid_face, live, score = self.liveness_checker.predict(
                    face_pil
                )
                if not valid_face:
                    continue
                
                # bounding box
                x, y, x2, y2 = [int(x) for x in box]
                bbox = [x, y, x2, y2]
                
                # recognition
                face_id = self._get_id(face_pil)
                
                # append
                recg_faces["faces"].append(np.array(face))
                recg_faces["face_ids"].append(face_id)
                recg_faces["livenesses"].append(live)
                recg_faces["liveness_scores"].append(score)
                recg_faces["bboxes"].append(bbox)

        return bool(recg_faces.get('faces')), recg_faces

                
    def _get_id(self, face:Image.Image) -> str:
        # encode face
        face_embedding = self.recognizer.encode(
            face
        )
        
        # comapre
        detect_dict = {}
        for k, v in self.embeddings.items():
            detect_dict[k] = (v - face_embedding).norm().item()
        min_key = min(detect_dict, key=detect_dict.get)
        recg_score = detect_dict[min_key]
        if recg_score >= self.recg_thresh:
            min_key = 'Undetected'
        
        return min_key