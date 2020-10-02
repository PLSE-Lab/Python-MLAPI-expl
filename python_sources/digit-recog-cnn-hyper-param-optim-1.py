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


#importing additional libraries

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# create data sets, skipping the header row with [1:]
dataset = pd.read_csv("../input/train.csv")
y_train = dataset[[0]].values.ravel()
X_train = dataset.iloc[:,1:].values
test = pd.read_csv("../input/test.csv").values

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
test = test.reshape(test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
test = test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)


X_train1 = X_train
y_train1 = y_train


#used later to specify the number of neurons in the output layer
num_classes = y_train1.shape[1]


def baseline_model2():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(16, (3, 3), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='sigmoid'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	return model

#for Kaggle submission

# Fit the model
model.fit(X_train1, y_train1, nb_epoch=10, batch_size=200, verbose=0)

# Prediction
pred = model.predict(test)
pred1 = pred.argmax(axis=1)
pred2 = np.c_[range(1,len(test)+1),pred1]

# save results
np.savetxt('submission_cnn_keras.csv', pred2, delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')