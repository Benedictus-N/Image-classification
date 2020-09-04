from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

x = pickle.load(open("x.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

epochs = 10

x = x/255.0

#using for loop to change 3 parameters using 3 different value and search the best settings out of the 3 different value for each parameter
dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            name = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(name)

            model = Sequential()
            model.add(Conv2D(layer_size, (3,3), input_shape = x.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

            history = model.fit(x, y, batch_size=32, epochs=epochs, validation_split=0.1)

            acc = history.history['acc']
            val_acc = history.history['val_acc']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs_range = range(epochs)

            plt.figure(figsize=(8,8))
            plt.subplot(1,2,1)
            plt.plot(epochs_range, acc, label='training accuracy')
            plt.plot(epochs_range, val_acc, label='validation accuracy')
            plt.legend(loc='lower right')
            plt.title('training and validation accuracy')

            plt.subplot(1,2,2)
            plt.plot(epochs_range, loss, label='training loss')
            plt.plot(epochs_range, val_loss, label='validation loss')
            plt.legend(loc='upper right')
            plt.title('training and validation loss')
            plt.show()