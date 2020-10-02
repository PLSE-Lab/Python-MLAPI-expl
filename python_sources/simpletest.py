# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils import np_utils
from keras.models import  Sequential
from keras.layers.core import  Lambda , Dense, Activation, Flatten, Dropout, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
xTrain = train.ix[:,1:].values.astype('float32')
xTest = test.values.astype('float32')
yTrain = np_utils.to_categorical(train.ix[:,0].values.astype('int32'))

inputDim = xTrain.shape[1]

model = Sequential()
#model.add(Dense(64, input_dim = inputDim))
model.add(Reshape((28, 28, 1), input_shape = (inputDim,)))
model.add(Conv2D(8, 3, activation='relu'))
model.add(Conv2D(8, 3, activation='relu'))
#model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.05))
#model.add(Dense(64))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
#model.add(Dropout(0.1))
model.add(Dense(10))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=["accuracy"])

model.fit(xTrain, yTrain, epochs=1, batch_size=16, validation_split=0.1)

predictions = model.predict_classes(xTest)

pd.DataFrame({"ImageId": list(range(1, len(predictions)+1)), "Label": predictions}).to_csv("output.csv", index = False, header=True)

# Any results you write to the current directory are saved as output.