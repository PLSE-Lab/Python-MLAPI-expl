# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.

print("loading data")

x_train = pd.read_csv("../input/emnist-letters-train.csv")
x_test = pd.read_csv("../input/emnist-letters-test.csv")
x_train = x_train.values
x_test = x_test.values

y_train = x_train[:,:1]
y_train = np.array(list(map(lambda x: x-1, y_train.flatten())))
x_train = x_train[:,1:]

y_test = x_test[:,:1]
y_test = np.array(list(map(lambda x: x-1, y_test.flatten())))
x_test = x_test[:,1:]
print("done")

print("transposing data:")
x_train = np.array(list(map(lambda x: x.reshape(28,28).transpose() ,x_train)))
x_test = np.array(list(map(lambda x: x.reshape(28,28).transpose() ,x_test)))
print("done")

"""
print("showing images:")
import matplotlib.pyplot as plt
plt.imshow(x_train[1].reshape(28,28))
"""

print("reshaping data:")
#for CNN
#x_train = np.array(list(map(lambda x: x.reshape(28,28,1) ,x_train)))
#x_test = np.array(list(map(lambda x: x.reshape(28,28,1) ,x_test)))

#for Dense
x_train = np.array(list(map(lambda x: x.flatten() ,x_train)))
x_test = np.array(list(map(lambda x: x.flatten() ,x_test)))

print("done")

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop, Adam

import random

batch_size = 256
num_classes = 26
epochs = 50

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation="relu", input_shape=(784,)))
model.add(Dropout(0.5))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))

"""
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation="relu",
                 input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
"""
model.add(Dense(num_classes, activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer=Adam(lr=0.001),
              metrics=["accuracy"])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_split=0.1)
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model.save("lettersDNN.h5")