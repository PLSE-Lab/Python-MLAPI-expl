from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

import pandas as pd
import numpy as np

# Read data
train = pd.read_csv('../input/train.csv')
labels = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (pd.read_csv('../input/test.csv').values).astype('float32')

# convert list of labels [0-9] to binary class matrix
y_train = np_utils.to_categorical(labels) 

# pre-processing: divide by max (255) and substract mean = range [-1,1]
scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.std(X_train)
X_train -= mean
X_test -= mean

# Images are 28px x 28px so input_dim = 784
input_dim = X_train.shape[1]

# Digits range from 0-9 so 10 classes
num_classes = y_train.shape[1]

def deep_dumb_network():
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(num_classes, activation='softmax'))

    # we'll use categorical xent for the loss, and RMSprop as the optimizer
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model
    
def larger_model():
  # create model
  model = Sequential()
  model.add(Convolution2D(30, 5, 5, border_mode= 'valid' , input_shape=(28, 28, 1),
      activation= 'relu' ))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Convolution2D(15, 3, 3, activation= 'relu' ))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(128, activation= 'relu' ))
  model.add(Dense(50, activation= 'relu' ))
  model.add(Dense(num_classes, activation= 'softmax' ))
  # Compile model
  model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
  return model

#model = deep_dumb_network()
model = larger_model()

print("Training...")
model.fit(X_train, y_train, nb_epoch=6, batch_size=16, validation_split=0.1, verbose=2)

print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=0)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "keras-mlp.csv")

