import numpy as np
from cfg.create_config import Config

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2


class Model:
    def __init__(self, X_train, y_train, X_valid, y_valid, weights_folder):
        self.model_cfg_path = "cfg\\syrex.json"
        self.model_weights_path = weights_folder + "\\syrex-{epoch:04d}-{val_acc:.2f}.weights"

        self.callbacks_list = [ModelCheckpoint(self.model_weights_path,
                                               monitor='val_acc',
                                               verbose=1,
                                               save_best_only=True,
                                               mode='max')]

        self.train_cfg = Config()
        self.cfg_data = self.train_cfg.load_cfg()

        self.num_features = self.cfg_data['num_features']
        self.num_labels = self.cfg_data['num_labels']
        self.batch_size = self.cfg_data['batch_size']
        self.epochs = self.cfg_data['epochs']
        self.width = self.cfg_data['width']
        self.height = self.cfg_data['height']

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

    def train(self):
        model = Sequential()

        model.add(Conv2D(self.num_features, kernel_size=(3, 3), activation='relu', input_shape=(self.width, self.height, 1),
                         data_format='channels_last', kernel_regularizer=l2(0.01)))
        model.add(Conv2D(self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(2 * self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(2 * self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(2 * 2 * self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(2 * 2 * self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(2 * 2 * 2 * self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(2 * 2 * 2 * self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(2 * 2 * 2 * self.num_features, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(2 * 2 * self.num_features, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(2 * self.num_features, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.num_labels, activation='softmax'))

        model.summary()

        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                      metrics=['accuracy'])

        model.fit(np.array(self.X_train), np.array(self.y_train),
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  callbacks=self.callbacks_list,
                  validation_data=(np.array(self.X_valid), np.array(self.y_valid)),
                  shuffle=True)

        model_cfg = model.to_json()
        with open(self.model_cfg_path, "w") as json_file:
            json_file.write(model_cfg)
        print("Training was successfully completed")
        print("Saved model to disk")
