
# Facial Recognition with Anti-Spoofing
The Facial Recognition with Anti-Spoofing is a system which capable of accurately identifying individuals while simultaneously detecting and preventing spoofing attempts. Spoofing attempts include using photographs, videos, or masks to deceive the facial recognition system.
## TODO
#### 1. fine-tune spoofNet
## Installation

#### 1. install packages with pip
```bash
pip install git+https://github.com/shijunjie07/facial_fas_recognition.git
```
#### 2. Download the pretrained weights from this [link](https://drive.google.com/drive/folders/1dqH2P7YGROh9SbjQDMsMX0vs8O8JzDPE?usp=sharing)
There are three weigths for dectector, recognizer, and anti-spoofing.
```bash
https://drive.google.com/drive/folders/1dqH2P7YGROh9SbjQDMsMX0vs8O8JzDPE?usp=sharing
```

#### 3. Put the pretrained weigths folder to the same directory
```bash
/pretrained
 - detector.pt
 - recognizer.pt
 - spoof.pt
```

#### 4. Change the 'pretrained_weights_dir' on config.yaml to your actual directory
```bash
pretrained_weights_dir: /path/to/your/pretrained/dir
```
## How To Use

```python
import cv2
from facial_fas_recognition from FaceRecg

face_recg = FaceRecg()

image = cv2.imread('/path/to/your/image/file')

is_not_empty, recg_data = face_recg.recg(image)

# output: recg_data:
{
    "faces": [np.array()],
    "face_ids": [str],
    "livenesses": [bool],
    "liveness_scores": [float],
    "bboxes": [list[int]],
}
```

#### how to change the 'pretrained_weights_dir' on your code
```python
from facial_fas_recognition.config import Config

config_instance = Config()
config_instance.set_directory('/path/to/your/new/pretrained/dir')

```