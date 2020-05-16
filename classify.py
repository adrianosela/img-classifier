#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

dim = 32
depth = 3
filter_width = 3
filter_height = 3
stride_width = 2
stride_height = 2
categories = 10

NUMBER_TO_CLASS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CACHE_FILENAME = 'trained_classifier.h5'

def load_and_prepare_data():
    """
    data from CIFAR. The dataset is split into 60k entries for training
    and 10k entries for testing. The images are 32 x 32 pixels in size
    and have a depth of 3 (for red, green, blue values).
    """
    (xtr, ytr), (xte, yte) = tf.keras.datasets.cifar10.load_data()
    x_train = xtr.astype('float32') / 255
    x_test = xte.astype('float32') / 255
    y_train = tf.keras.utils.to_categorical(ytr, categories)
    y_test = tf.keras.utils.to_categorical(yte, categories)
    return x_train, y_train, x_test, y_test

def image_recognition_cnn():
    model = tf.keras.models.Sequential([
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, saveto):
    hist = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_split=0.2)
    model.save(saveto)
    return hist

def evaluate_model(model, x_test, y_test):
    return model.evaluate(x_test, y_test)

def visualize_loss(hist):
    plt.title('Model Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()

def visualize_accuracy(hist):
    plt.title('Model Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()

def predict(model, image):
    probabilities = model.predict(np.array( [image,] ))
    index = np.argsort(probabilities[0,:])
    print("Most likely class:", NUMBER_TO_CLASS[index[9]], "-- Probability:", probabilities[0,index[9]])
    print("Second most likely class:", NUMBER_TO_CLASS[index[8]], "-- Probability:", probabilities[0,index[8]])
    print("Third most likely class:", NUMBER_TO_CLASS[index[7]], "-- Probability:", probabilities[0,index[7]])
    print("Fourth most likely class:", NUMBER_TO_CLASS[index[6]], "-- Probability:", probabilities[0,index[6]])
    print("Fifth most likely class:", NUMBER_TO_CLASS[index[5]], "-- Probability:", probabilities[0,index[5]])

if __name__ == "__main__":
    x_tr, y_tr, x_te, y_te = load_and_prepare_data()

    if os.path.exists(CACHE_FILENAME):
        model = tf.keras.models.load_model(CACHE_FILENAME)
    else:
        model = image_recognition_cnn()
        hist = train_model(model, x_tr, y_tr, cache)
        visualize_loss(hist)

    # call predict with any 32 x 32 x 3 image
