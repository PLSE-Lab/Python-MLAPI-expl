#Code written in Python 2.7. Kaggle score for submission 0.98043. Code does not run on Python 3.6.
#Code and ideas borrowed from machinelearningmastery.com
#Following hyper parameters changed:
#1. Addition of Convolution and Pooling layers with new filter and window size settings
#2. Dense layer activation function changed from Relu to Sigmoid
#3. Optimizer changed from ADAM to rmsprop

import numpy as np # linear algebra
import pandas as pd

#importing additional libraries

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# create data sets, skipping the header row with [1:]
dataset = pd.read_csv("/home/bala/Documents/KaggleDigitRecognition/train.csv")
y_train = dataset[[0]].values.ravel()
X_train = dataset.iloc[:,1:].values
test = pd.read_csv("/home/bala/Documents/KaggleDigitRecognition/test.csv").values

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
test = test.reshape(test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
test = test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)


X_train1 = X_train
y_train1 = y_train


#used later to specify the number of neurons in the output layer
num_classes = y_train1.shape[1]


def baseline_model2():
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16, 3, 3, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model
#model = baseline_model2()
 #Convolution2D
 #model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
 #model.add(Conv2D(16, (3, 3), input_shape=(1, 28, 28), activation='relu'))
#for Kaggle submission

# choosing a random subset of the available data to train on. 'X' has image data & 'y' has the label
A=X_train.shape[0]
B=np.random.randint(A,size=20000)
X_train1=X_train[B,:]
y_train1=y_train[B,:]

#C=X_test.shape[0]
#D=np.random.randint(C,size=1000)
#X_test1=X_test[D,:]
#y_test1=y_test[D,:]

C=test.shape[0]
D=np.random.randint(C,size=1000)
X_test1=test[D,:]
y_test1=test[D,:]

# build the model
model = baseline_model2()
model.fit(X_train1, y_train1, nb_epoch=10, batch_size=200, verbose=0)

# Prediction
pred  = model.predict(test)
pred1 = pred.argmax(axis=1)
pred2 = np.c_[range(1,len(test)+1),pred1]

# save results
np.savetxt('/home/bala/Documents/KaggleDigitRecognition/submission_cnn_keras.csv', pred2, delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')