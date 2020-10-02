from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
import numpy as np

# Read data
train = pd.read_csv('../input/train.csv')
labels = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (pd.read_csv('../input/test.csv').values).astype('float32')

# convert list of labels to binary class matrix
y_train = np_utils.to_categorical(labels) 

# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.std(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

# Here's a Deep Dumb MLP (DDMLP)
model = Sequential()
model.add(Dense(input_dim, 128, init='lecun_uniform', activation='relu'))
model.add(Dense(128, 128, init='lecun_uniform', activation='relu'))
model.add(Dense(128, nb_classes, init='lecun_uniform', activation='relu'))

# we'll use MSE (mean squared error) for the loss, and RMSprop as the optimizer
model.compile(loss='mse', optimizer='rmsprop')

print("Training...")
model.fit(X_train, y_train, nb_epoch=10, batch_size=16, show_accuracy=False, verbose=1)

print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=0)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "keras-mlp.csv")
