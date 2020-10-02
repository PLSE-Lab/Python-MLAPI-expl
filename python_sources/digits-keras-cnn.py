# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


import keras
import numpy as np
import pandas as pd
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.core import Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


print("1. loading data")
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

Y_train = train.values[:,0]
X_train = train.values[:,1:]
X_test = test.values

y_train = to_categorical(Y_train)
x_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
x_test = X_test.reshape(test.shape[0], 28, 28, 1)



print("2. running cnn")
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
        activation='relu',
        input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=['accuracy'])
model.fit(x_train, y_train,
        batch_size=128,
        validation_split=0.1)

pred = model.predict(x_test)

print(pred)
