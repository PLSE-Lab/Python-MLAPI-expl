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

# -*- coding: utf-8 -*-
"""
Created on Wed May 17 18:23:31 2017

@author: Amajid Sinar
"""

import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("ggplot")
import numpy as np

#Importing training set
training_set = pd.read_csv("../input/exoTrain.csv")
X_train = training_set.iloc[:,1:].values
y_train = training_set.iloc[:,0:1].values

#Importing test set
test_set = pd.read_csv("../input/exoTest.csv")
X_test = test_set.iloc[:,1:].values
y_test = test_set.iloc[:,0:1].values

#Scale the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Convert data into 3d tensor
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))


#Importing convolutional layers
from keras.models import Sequential
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization

#Convolution steps
#1.Convolution
#2.Max Pooling
#3.Flattening
#4.Full Connection

#Initialising the CNN
classifier = Sequential()

#Input shape must be explicitly defined, DO NOT USE (None,shape)!!!
#1.Multiple convolution and max pooling
classifier.add(Convolution1D(filters=8, kernel_size=11, activation="relu", input_shape=(3197,1)))
classifier.add(MaxPooling1D(strides=4))
classifier.add(BatchNormalization())

#2.Flattening
classifier.add(Flatten())

#3.Full Connection
classifier.add(Dropout(0.5))
classifier.add(Dense(64, activation='relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(64, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

#Configure the learning process
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#Train!
classifier.fit(X_train, y_train, batch_size=32, epochs=1, validation_data=(X_test,y_test))

score = classifier.evaluate(X_test, y_test)
#Performance = 99.1%