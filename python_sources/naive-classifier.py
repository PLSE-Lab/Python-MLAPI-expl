# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

print(os.listdir("../input"))

test_size = 400

def read_data():
    allFiles = glob.glob("../input/*.csv")
    list = []
    for file in allFiles:
        df = pd.read_csv(file,index_col=None)
        read_matrix = np.asmatrix(df)
        list.append(read_matrix)
    data = np.concatenate(list)
    values = np.asmatrix(data)
    np.random.shuffle(values)
    x_train = values[test_size:,:64]
    y_train = values[test_size:,64]
    x_test = values[:test_size,:64]
    y_test = values[:test_size,64]
    print('x_train ', x_train.shape)
    print('y_train ', y_train.shape)
    print('x_test ', x_test.shape)
    print('y_test ', y_test.shape)
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = read_data()

y_train = keras.utils.to_categorical(y_train, num_classes=4)
y_test = keras.utils.to_categorical(y_test, num_classes=4)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
model.add(Dense(36, activation='relu', input_dim=64))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=100)
score = model.evaluate(x_test, y_test, batch_size=test_size)
print(score)