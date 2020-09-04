#%%
from __future__ import absolute_import, division, print_function, unicode_literals

#import useful package
import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
#%%

#load the training data from load_dataset_original.py
x = pickle.load(open("x.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

#normalize the image
x = x/255.0

#define the number of epoch to be used during the model fitting and graphic plotting
epochs = 100

#using sequential model type and couple of layers
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation('sigmoid'))

#compiling model using adam optimizer and binary_crossentropy loss function
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

#define batch size and split image number 10% for validation purpose
history = model.fit(x, y, batch_size=32, epochs=epochs, validation_split=0.1)

#determine the variable to be included in the graphic
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

#define the graph
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

#%%
