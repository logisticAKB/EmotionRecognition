import pandas as pd
import numpy as np
from cfg.create_config import Config
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from model.train_model import Model


def parse_args():
    parser = ArgumentParser(description='Start training')
    parser.add_argument('--data', dest='data', default='data/train.csv', type=str)
    parser.add_argument('--weights_folder', dest='weights_folder',
                        default='weights', type=str)

    args = parser.parse_args()
    return args


def prepare_data(path):
    print("Starting Preprocessing...")

    config = Config()
    cfg = config.load_cfg()
    width, height = cfg['width'], cfg['height']

    data = pd.read_csv(path)

    datapoints = data['pixels'].tolist()

    X = []
    for xseq in datapoints:
        xx = [int(xp) for xp in xseq.split(' ')]
        xx = np.asarray(xx).reshape(width, height)
        X.append(xx.astype('float32'))

    X = np.asarray(X)
    X = np.expand_dims(X, -1)

    y = pd.get_dummies(data['target']).as_matrix()

    print("Preprocessing Done!")
    print("Number of Features: " + str(len(X[0])))
    print("Number of Labels: " + str(len(y[0])))
    print("Number of examples in dataset:" + str(len(X)))

    return X, y


if __name__ == '__main__':
    args = parse_args()

    x, y = prepare_data(args.data)

    x -= np.mean(x, axis=0)
    x /= np.std(x, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

    model = Model(X_train, y_train, X_valid, y_valid, args.weights_folder)
    model.train()
