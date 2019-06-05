import cv2
import numpy as np
from keras.models import model_from_json
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description='Detect emotions')
    parser.add_argument('--image', dest='image', default='data/test.jpg', type=str)
    parser.add_argument('--labels', dest='labels', default='data/data.labels', type=str)
    parser.add_argument('--model', dest='model', default='cfg/syrex.json', type=str)
    parser.add_argument('--weights', dest='weights',
                        default='weights/syrex.weights', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    with open(args.model, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    model.load_weights(args.weights)
    print("Loaded model from disk")

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    with open(args.labels, 'r') as labels:
        target = []
        for label in labels.readlines():
            target.append(label.strip())

    font = cv2.FONT_ITALIC  # FONT_HERSHEY_SIMPLEX

    im = cv2.imread(args.image)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2, 5)
        face_crop = im[y:y + h, x:x + w]

        face_crop = cv2.resize(face_crop, (48, 48))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_crop = [face_crop.astype('float32')]

        face_crop = np.asarray(face_crop)

        face_crop = np.expand_dims(face_crop, -1)

        result = target[np.argmax(model.predict(face_crop).tolist())]

        cv2.putText(im, result, (x, y + h + 20), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('result', im)
    # cv2.imwrite('4.jpg', im)
    cv2.waitKey(0)
