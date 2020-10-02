
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


# load_data
train_data = pd.read_csv('../input/train.csv')

labels = train_data.label.values.astype('int32')
train_data = train_data.drop('label', axis=1).as_matrix().astype('float32')

test_data = pd.read_csv('../input/test.csv').as_matrix().astype('float32')

# make label vectors
y_train = np_utils.to_categorical(labels) 

# normalize data
X_train = train_data / np.max(train_data)
X_test = test_data / np.max(train_data)


img_size = (28, 28)

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_size[0], img_size[1])
    X_test = X_test.reshape(X_test.shape[0], 1, img_size[0], img_size[1])
    image_shape = (1, img_size[0], img_size[1])
else:
    X_train = X_train.reshape(X_train.shape[0], img_size[0], img_size[1], 1)
    X_test = X_test.reshape(X_test.shape[0], img_size[0], img_size[1], 1)
    image_shape = (img_size[0], img_size[1], 1)


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print("Build a CNN")
model = Sequential()

model.add(Convolution2D(8, 3, 3, border_mode='valid', input_shape=image_shape))
model.add(Activation('relu'))
model.add(Convolution2D(12, 3, 3, border_mode='valid', input_shape=image_shape))
model.add(Activation('relu'))
model.add(Convolution2D(12, 4, 4, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(48))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


print("Training...")
model.fit(X_train, y_train, nb_epoch=20, batch_size=64, verbose=2)

print("Predictions...")
predictions = model.predict_classes(X_test, verbose=0)

preds = pd.DataFrame({"Label": predictions})
preds.index = preds.index + 1
preds.to_csv("keras_cnn02.csv",index_label='ImageId')

print ("That's all.")



