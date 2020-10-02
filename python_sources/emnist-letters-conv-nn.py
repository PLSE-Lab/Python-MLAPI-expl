import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


import os

batch_size = 128
num_classes = 26
epochs = 10

# balanced_train_path = '../input/emnist-balanced-train.csv'
# balanced_test_path = '../input/emnist-balanced-test.csv'
# byclass_train_path = '../input/emnist-byclass-train.csv'
# byclass_test_path = '../input/emnist-byclass-test.csv'
# bymerge_train_path = '../input/emnist-bymerge-train.csv'
# bymerge_test_path = '../input/emnist-bymerge-test.csv'
# digits_train_path = '../input/emnist-digits-train.csv'
# digits_test_path = '../input/emnist-digits-test.csv'
letters_train_path = '../input/emnist-letters-train.csv'
letters_test_path = '../input/emnist-letters-test.csv'
# mnist_train_path = '../input/emnist-mnist-train.csv'
# mnist_test_path = '../input/emnist-mnist-test.csv'

# balanced_train_data = pd.read_csv(balanced_train_path)
# balanced_test_data = pd.read_csv(balanced_test_path)
# byclass_train_data = pd.read_csv(byclass_train_path)
# byclass_test_data = pd.read_csv(byclass_test_path)
# bymerge_train_data = pd.read_csv(bymerge_train_path)
# bymerge_test_data = pd.read_csv(bymerge_test_path)
# digits_train_data = pd.read_csv(digits_train_path)
# digits_test_data = pd.read_csv(digits_test_path)
letters_train_data = pd.read_csv(letters_train_path)
letters_test_data = pd.read_csv(letters_test_path)
# mnist_train_data = pd.read_csv(mnist_train_path)
# mnist_test_data = pd.read_csv(mnist_test_path)

print('data loaded')

datas_train = letters_train_data.values
datas_test = letters_test_data.values

minusOne = lambda x: x - 1

x_train = datas_train[:, 1:].astype('float32')
y_train = datas_train[:, 0:1]
y_train = np.array(list(map(minusOne, y_train)))

x_test = datas_test[:, 1:].astype('float32')
y_test = datas_test[:, 0:1]
y_test = np.array(list(map(minusOne, y_test)))

x_train /= 255
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

print('Compiling the model...')

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
              
print('Let\'s the training  begins !')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('LETTERS_CNN.h5')

