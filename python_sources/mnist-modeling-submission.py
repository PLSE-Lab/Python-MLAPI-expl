# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 15:06:39 2017

@author: katsuhisa
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# import further libraries
import matplotlib.pyplot as plt
import seaborn as sns

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# load training & test datasets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# pandas to numpy
y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)

del train

# normalize
X_train = X_train/255.0
test = test/255.0

# reshape the data so that the data
# represents (label, img_rows, img_cols, grayscale)
X_train = X_train.values.reshape(-1, 28, 28, 1)
test = test.values.reshape(-1, 28, 28, 1)

# one-hot vector as a label (binarize the label)
y_train = to_categorical(y_train, num_classes=10)

# convolutional 2D NN
model = Sequential()
model.add(Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')

# check input and output shapes
print("input shape: ", model.input_shape)
print("input shape: ", model.output_shape)
print("X_train shape: ", X_train.shape)
print("test shape: ", test.shape)

# cross validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=1220)

# prevent overfitting (data argumentation)
gen = image.ImageDataGenerator()
gen.fit(X_train)

# learning rate
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

# fit model to the training data
history = model.fit_generator(gen.flow(X_train, y_train,
                                       batch_size=36), epochs=30, validation_data = (X_val, y_val),
                                       verbose=2, steps_per_epoch=X_train.shape[0]/36,
                                       callbacks=[learning_rate_reduction])

# model prediction on test data
predictions = model.predict_classes(test, verbose=0)

# submission
submissions = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
    "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)
