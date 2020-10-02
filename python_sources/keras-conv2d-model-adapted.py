import keras
import pandas as pd
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split

batch_size = 128
num_classes = 10
epochs = 15


def load_kaggle(F):
    df = pd.read_csv(F)
    labels = df['label']
    pixels = df.drop('label', axis=1)
    return np.array(pixels).reshape(-1, 28, 28, 1), np.array(labels)

def load_test(F):
    df = pd.read_csv(F)
    pixels = df.drop("ImageId", axis=1)
    ids = df['ImageId']
    return np.array(pixels).reshape(-1, 28, 28, 1), np.array(ids)

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
feature_data, label_data = load_kaggle("../input/train.csv")
x_train, x_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.2)
# This is the same MNIST dataset, but consumes less memory
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

del x_train, y_train, x_test, y_test

X_test, Ids = load_test("../input/test.csv")
labels = np.array(model.predict(X_test), dtype=np.int23)
result = pd.DataFrame({
    'ImageId':np.array(Ids),
    'Label':labels
})
result.to_csv("result.csv", index=False)