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


## the code is adapted from fchollet/keras
# link: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

np.random.seed(42)  # for reproducibility

# ## 1. Import library
from keras.models import Sequential
import keras.layers.core as core
import keras.models as models
import keras.utils.np_utils as kutils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD

# ## 2. Load data
train = pd.read_csv("../input/train.csv").values
test  = pd.read_csv("../input/test.csv").values

# input image dimensions
img_rows, img_cols = 28, 28
X_train = train[:, 1:].reshape(train.shape[0], 1, img_rows, img_cols)
X_test = test.reshape(test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = kutils.to_categorical(train[:, 0])

print('X_train ',X_train.shape)
print('X_test ',X_test.shape)
print('Y_train ',Y_train.shape)

# build model
nb_classes = 10
batch_size = 512
nb_epoch = 1

# number of convolutional filters to use
nb_filters = 3
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
# use optimizer
#model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

# train model
model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size)
# evaluate 
#loss_and_metrics = model.evaluate(X_train, Y_train, batch_size=256)
#print('loss_and_metrics',loss_and_metrics)

# save into file
Y_test = model.predict_classes(X_test)
np.savetxt('mnist_submit.csv', np.c_[range(1,len(Y_test)+1),Y_test], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')