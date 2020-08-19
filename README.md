<p align="center">
<img align="center" alt="sample" src="https://github.com/logisticAKB/EmotionRecognition/blob/master/data/example.jpg" />
</p>

# EmotionRecognition - Show your expression (syrex)

This is a Python program that detects faces in images/webcam live feed through haarcascade and recognizes emotion using Convolutional Neural Network.

## Installation

- Install required libraries via [pip](https://pip.pypa.io/en/stable/)
  ```bash
  pip install -r requirements.txt
  ```
- Download [syrex.weights](https://drive.google.com/file/d/1WUncPDIa1CDv0dHck1rv4a-mCPWYnWf1/view?usp=sharing) file and put it into ```EmotionRecognition/weights/``` folder

## Usage

```bash
# emotion recognition on images
python3 detect.py \
  --image=data/test.jpg \
  --labels=data/data.labels \
  --model=cfg/syrex.json \
  --weights=weights/syrex.weights 
  
# emotion recognition on live feed from webcam
python3 detect_realtime.py \
  --labels=data/data.labels \
  --model=cfg/syrex.json \
  --weights=weights/syrex.weights
```

## Training on your data

1. Prepare your data
    - Put your images to the ```EmotionRecognition/data/train/images/``` folder
    - Put your labels to the ```EmotionRecognition/data/train/labels/``` folder
#### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Note: filenames of images must match labels filenames
2. In order to create csv file needed for training run:
    ```bash
    # from EmotionRecognition/scripts/
    python3 image2pixels.py \
      --image_folder=data/train/images \
      --label_folder=data/train/labels \
      --output=data/train.csv \
      --labels_decode=data/data.labels
    ```
3. Set train parameters in ```EmotionRecognition/cfg/train.cfg``` file
4. Run training:
```bash
python3 train.py \
  --data=data/train.csv \
  --weights_folder=weights/
```

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
