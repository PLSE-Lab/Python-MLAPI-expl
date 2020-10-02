import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

np.random.seed(1234)

train = pd.read_csv("../input/train.csv").values
test = pd.read_csv("../input/test.csv").values

x_train = train[:, 1:]
y_train = train[:, 0]
x_test = test

num_pixels = x_train.shape[1]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train)
num_classes = y_train.shape[1]

model = Sequential()
model.add(Dense(num_pixels / 2, input_dim=num_pixels, init='normal', activation='relu'))
model.add(Dense(num_classes, init='normal', activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, nb_epoch=10, batch_size=200, verbose=2)

predictions = model.predict_classes(x_test)

np.savetxt('hello_mnist.csv',
           np.c_[range(1, len(predictions)+1), predictions],
           delimiter=',',
           header='ImageId,Label',
           comments='',
           fmt='%d')
