#!/usr/bin/env python

import tensorflow as tf
import matplotlib.pyplot as plt

dim = 32
depth = 3
filter_width = 3
filter_height = 3
stride_width = 2
stride_height = 2
categories = 10

cache = 'model.h5'

class Model():

    def __init__(self):
        """
        initialize the model by downloading the training and testing
        data from CIFAR. The dataset is split into 60k entries for training
        and 10k entries for testing. The images are 32 x 32 pixels in size
        and have a depth of 3 (for red, green, blue values).
        """
        (xtr, ytr), (xte, yte) = tf.keras.datasets.cifar10.load_data()
        self.x_train = xtr.astype('float32') / 255
        self.x_test = xte.astype('float32') / 255
        self.y_train = tf.keras.utils.to_categorical(ytr, categories)
        self.y_test = tf.keras.utils.to_categorical(yte, categories)

    def load(self, filename):
        self.model = tf.keras.models.load_model(filename)

    def build(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(dim, (filter_width, filter_height), activation=tf.nn.relu, padding='same', input_shape=(dim, dim, depth)),
            tf.keras.layers.Conv2D(dim, (filter_width, filter_height), activation=tf.nn.relu, padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(stride_width, stride_height)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(dim*2, (filter_width, filter_height), activation=tf.nn.relu, padding='same', input_shape=(dim, dim, depth)),
            tf.keras.layers.Conv2D(dim*2, (filter_width, filter_height), activation=tf.nn.relu, padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(stride_width, stride_height)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(dim*16, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(categories, activation=tf.nn.softmax),
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self):
        self.hist = self.model.fit(self.x_train, self.y_train, batch_size=32, epochs=20, validation_split=0.2)
        self.model.save(cache)

    def visualize_loss(self):
        plt.title('Model Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(self.hist.history['loss'])
        plt.plot(self.hist.history['val_loss'])
        plt.legend(['training', 'validation'], loc='upper right')
        plt.show()

    def visualize_accuracy(self):
        plt.title('Model Accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.plot(self.hist.history['accuracy'])
        plt.plot(self.hist.history['val_accuracy'])
        plt.legend(['training', 'validation'], loc='upper right')
        plt.show()

if __name__ == "__main__":
    m = Model()
    if os.path.exists(cache):
        m.load_cached(cache)
    else:
        m.build()
        m.train()
    m.visualize_loss()
