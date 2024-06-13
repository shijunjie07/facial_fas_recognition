# ---------------------------------------------------
# embedding generation
# @author: Shi Junjie
# Tue 11 June 2024
# ---------------------------------------------------

import os
import cv2
from PIL import Image
from tqdm import tqdm

from .detector import FaceDetector
from .recognizer import Recognizer

# init facial recg
face_detector = FaceDetector()
encode = Recognizer().encode


def generate_embedding(image):
    
    _, cropped = face_detector.detect_faces(image)
    if cropped is not None:
        cropped = Image.fromarray(cropped[0])
        embedding = encode(cropped)[0, :]
        
        return True, embedding

    else: return False, False

# generate embeddings
def generate_known_face_embeddings(known_face_dir):
    
    embeddings = {}

    files = os.listdir(known_face_dir)
    for file in tqdm(files, desc="Generating Embeddings for Known Faces"):
        face_id, _ = file.split(".")

        img = cv2.imread(os.path.join(known_face_dir, file))
        _, cropped = face_detector.detect_faces(img)
        if cropped is not None:
            cropped = Image.fromarray(cropped[0])
            embeddings[face_id] = encode(cropped)[0, :]

    return embeddings