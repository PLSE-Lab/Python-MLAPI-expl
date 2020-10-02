import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import np_utils


def load_mnist(path, kind='train'):
    import os
    import numpy as np
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbfile:
        labels = np.frombuffer(lbfile.read(), dtype=np.uint8, offset=8)
    with open(images_path, 'rb') as imgfile:
        images = np.frombuffer(imgfile.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels


DATA_DIR = '../input'

X_train, Y_train = load_mnist(path=DATA_DIR, kind='train')
# X_train = X_train[:1000]
# Y_train = Y_train[:1000]
Y_train = np_utils.to_categorical(Y_train)
X_test, Y_test_classes = load_mnist(path=DATA_DIR, kind='t10k')
Y_test = np_utils.to_categorical(Y_test_classes)

X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(Y_train.shape[1], activation="softmax"))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=RMSprop(), metrics=['accuracy'])
# model.summary()

epochs = 10

model.fit(X_train, Y_train, batch_size=128, epochs=epochs, validation_data=(X_test, Y_test), verbose=2, shuffle=True)
