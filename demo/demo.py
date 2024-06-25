# -----------------------------
# facial fas recognition demo
# @author: Shi Junjie
# Tue 25 June 2024
# -----------------------------

import cv2
import time
import numpy as np
import warnings
warnings.filterwarnings('ignore')


from facial_fas_recognition.face_recg import FaceRecg
from facial_fas_recognition.embd import generate_known_face_embeddings


# generate embeddings for multiple images
known_face_dir = 'path/to/yout/face/dir'
embeddings = generate_known_face_embeddings(known_face_dir)


def resize_with_aspect_ratio(image, target_size):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Determine the scale factor and new dimensions
    if width > height:
        new_width = target_size
        new_height = int(target_size * height / width)
    else:
        new_height = target_size
        new_width = int(target_size * width / height)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def add_black_padding(image, padding_width):
    # Get the dimensions of the resized image
    height, width = image.shape[:2]
    
    # Create a new image with padding
    new_width = width + padding_width
    padded_image = np.zeros((height, new_width, 3), dtype=np.uint8)
    
    # Place the original image onto the new image
    padded_image[:, padding_width:] = image
    return padded_image

# create facial recognition instance
face_recg = FaceRecg(embeddings, 0.65)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()
    
padding_width = 0
last_frame_time = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        continue

    start_time = time.time()

    frame = resize_with_aspect_ratio(frame, 640)
    padded_frame = add_black_padding(frame, padding_width)

    is_not_empty, recg_data = face_recg.recg(frame)

    if is_not_empty:
        for i in range(len(recg_data['face_ids'])):
            x, y, x2, y2 = recg_data['bboxes'][i]
            
            box_color = (0, 0, 255)
            if recg_data['livenesses'][i]:
                box_color = box_color = (0, 255, 0)
            # draw bounding box
            cv2.rectangle(padded_frame, (x+padding_width, y), (x2+padding_width, y2), box_color, 2)
            cv2.putText(
                padded_frame, recg_data['face_ids'][i], (x + 5+padding_width, y+10), 
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(
                padded_frame, 'Real: {}'.format(recg_data['livenesses'][i][0, 0]), (10, 70), 
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(
                padded_frame, 'live score: {:.2f}'.format(recg_data['liveness_scores'][i]), (10, 90),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    # calculate fps
    end_time = time.time()
    diff_time = end_time - last_frame_time
    last_frame_time = end_time
    if diff_time == 0:
        fps = '-1'
    else:
        fps = round(1 / (diff_time), 2)
    
    processing_time = end_time - start_time
    
    # display fps
    cv2.putText(
        padded_frame, 'FPS: {}'.format(fps), (10, 30),
        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(
        padded_frame, 'Proc Time: {}s'.format(round(processing_time, 2)), (10, 50),
        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(
        padded_frame, 'Frame Size: {}'.format(frame.size), (100, 30),
        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Camera', padded_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()