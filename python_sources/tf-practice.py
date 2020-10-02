# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras import layers
import pathlib
import cv2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

labels = os.listdir("../input/cell_images/cell_images")
s = "../input/cell_images/cell_images/"
# listOfData = os.listdir(dataPath)
dataRoot = pathlib.Path()

# Any results you write to the current directory are saved as output.
labNum = [0,1]
nameMap = dict(zip(labels,labNum))
xInput_64 = []
xInput_128 = []
yInput = []
for name in nameMap:
    lab = nameMap[name]
    for img in os.listdir(s+name):
        try:
            xInput_64.append(cv2.resize(cv2.imread(s+name+"/"+img), (64,64)))
            xInput_128.append(cv2.resize(cv2.imread(s+name+"/"+img), (128,128)))
            yInput.append(lab)
        except:
            continue
yInput = tf.keras.utils.to_categorical(yInput, num_classes=2)
# print(len(xInput_64))
# print(len(xInput_128))
def modelMake():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64,3,(1,1), activation="relu", input_shape=(64,64,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128,3,(1,1), activation="relu"))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(256,3,(1,1), activation="relu"))
    model.add(layers.MaxPool2D())
    model.add(layers.Conv2D(512,3,(1,1), activation="relu"))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(0.4))
    model.add(layers.Conv2D(256,3,(1,1), activation="relu"))
    model.add(layers.MaxPool2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(2, activation="softmax"))
    return model
    
# orgInputs = tf.keras.Input(shape=(64,64,3), name="input")
# x = layers.Conv2D(64,3,(1,1), activation="relu")(orgInputs)
# x = layers.BatchNormalization()(x)
# x = layers.Conv2D(128,3,(1,1), activation="relu")(x)
# x = layers.MaxPool2D()(x)
# x = layers.Dropout(0.2)(x)
# x = layers.Conv2D(256,3,(1,1), activation="relu")(x)
# x = layers.MaxPool2D()(x)
# x = layers.Conv2D(512,3,(1,1), activation="relu")(x)
# x = layers.MaxPool2D()(x)
# x = layers.Dropout(0.4)(x)
# x = layers.Conv2D(256,3,(1,1), activation="relu")(x)
# x = layers.MaxPool2D()(x)
# x = layers.Flatten()(x)
# x = layers.Dense(64, activation="relu")(x)
# output = layers.Dense(2, activation="softmax")(x)
model = modelMake()
model.summary()

# n = model(inputs=orgInputs, outputs=output)
# print(type(model))

# model = tf.keras.Model(inputs=orgInputs, outputs=output)
# model.summary()
optimizer = tf.keras.optimizers.Adam(lr=0.001, decay=0.00001)
model.compile(optimizer, loss="categorical_crossentropy",metrics=['accuracy'])
# # print(xInput[0].shape)
run = model.fit(np.array(xInput_64),np.array(yInput),epochs=10,batch_size=512,validation_split=0.3)





    
