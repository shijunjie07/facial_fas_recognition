
# [DEV] Facial Recognition with Anti-Spoofing

The Facial Recognition with Anti-Spoofing is a system which capable of accurately identifying individuals while simultaneously detecting and preventing spoofing attempts. Spoofing attempts include using photographs, videos, or masks to deceive the facial recognition system.
## TODO
#### 1. Embeddings Generation
#### 2. README.md
## Installation

#### 1. install packages with pip
```bash
pip install git+https://github.com/shijunjie07/facial_fas_recognition.git
```
#### 2. Download the pretrained weights from this [link](https://drive.google.com/drive/folders/1dqH2P7YGROh9SbjQDMsMX0vs8O8JzDPE?usp=sharing)
#### 3. put the pretrained weigths folder to your directory
```bash
/pretrained
  - detector.pt
  - recognizer.pt
  - spoof.pt
```
#### 4. generate embedding for known faces
```python
face_recg = FaceRecg()
...

```
### 5. 
## Usage/Examples

```python
from facial_fas_recognition from FaceRecg

face_recg = FaceRecg()
```


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Acknowledgements