import numpy as np
import pandas as pd 
import os
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import *
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/digit-recognizer/train.csv')
test  = pd.read_csv('../input/digit-recognizer/test.csv')
df_features = train.iloc[:, 1:785]
df_label = train.iloc[:, 0]
x_test = test.iloc[:, 0:784]
x_train, x_cv, y_train, y_cv = train_test_split(df_features, df_label, 
                                                test_size = 0.2,
                                                random_state = 1212)
x_train = x_train.values.reshape(33600,28,28,1)
x_cv = x_cv.values.reshape(8400,28,28,1)
x_test = x_test.values.reshape(28000,28,28,1)
x_train = x_train.astype('float32'); x_cv= x_cv.astype('float32'); x_test = x_test.astype('float32')
x_train /= 255; x_cv /= 255; x_test /= 255
num_digits = 10
y_train = keras.utils.to_categorical(y_train, num_digits)
y_cv = keras.utils.to_categorical(y_cv, num_digits)
batch_size = 128
epochs = 20
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),
                      padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_digits, activation='softmax'))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
history=model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_cv, y_cv))