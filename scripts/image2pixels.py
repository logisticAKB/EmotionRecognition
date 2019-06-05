import os
import sys
import cv2
from argparse import ArgumentParser

import pandas as pd


def parse_args():
    parser = ArgumentParser(description='Converts images to pixels and store it in a csv file')
    parser.add_argument('--image_folder', dest='image_folder',
                        default='data/train/images/', type=str)
    parser.add_argument('--label_folder', dest='label_folder',
                        default='data/train/labels/', type=str)
    parser.add_argument('--output', dest='output',
                        default='data/train.csv', type=str)
    parser.add_argument('--labels_decode', dest='labels_decode', default='data/data.labels', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')

    if not os.path.exists(args.output):
        csv = pd.DataFrame(columns=['id', 'pixels', 'target'])
        csv.to_csv('../' + args.output, header=True, index=False)

    csv_file = pd.read_csv('../' + args.output)

    images = os.listdir(path='../' + args.image_folder)
    labels = os.listdir(path='../' + args.label_folder)

    if len(images) != len(labels):
        print("Error! The number of images must be equal to the number of labels.")
        sys.exit(1)

    id = csv_file.shape[0]
    for i in range(len(images)):
        path_to_image = '../' + args.image_folder + images[i]
        path_to_label = '../' + args.label_folder + labels[i]

        img = cv2.imread(path_to_image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1)

        (x, y, w, h) = faces[0]
        face_crop = gray[y:y + h, x:x + w]
        face_crop = cv2.resize(face_crop, (48, 48))

        pixels = ""
        for row in face_crop:
            for el in row:
                pixels += str(el) + ' '

        base_image = os.path.basename(path_to_image)
        base_label = os.path.basename(path_to_label)
        if os.path.splitext(base_image)[0] != os.path.splitext(base_label)[0]:
            print("Error! Name of the label must be equal to the name of the image.")
            sys.exit(1)

        with open('../' + args.labels_decode, 'r') as label_decoder:
            targets = {}
            code = 0
            for label in label_decoder.readlines():
                targets[label.strip()] = code
                code += 1

        with open(path_to_label, 'r') as label:
            target = targets[label.readline().strip()]

        csv_file = csv_file.append({'id': id, 'pixels': pixels.strip(), 'target': target}, ignore_index=True)
        id += 1

    csv_file.to_csv('../' + args.output, header=True, index=False)
