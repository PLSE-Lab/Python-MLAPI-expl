# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe, rand
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras import optimizers

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout


import numpy as np
i=X_train=np.load("../input/X_train"+ '.npy')
print(i.shape)
i=Y_train=np.load("../input/Y_train"+ '.npy')
print(i.shape)
i=X_test=np.load("../input/X_test"+ '.npy')
print(i.shape)
i=Y_test=np.load("../input/Y_test"+ '.npy')
print(i.shape)
epochs = 30
batch_size = 16
n_hidden = 32
timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
#n_classes = _count_classes(Y_train)
n_classes=6

def data():
    return X_train,X_test,Y_train,Y_test
    
def model(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(LSTM(n_hidden,return_sequences=True, input_shape=(timesteps, input_dim)))

    model.add(LSTM(n_hidden))

    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(Dense(n_classes, activation='sigmoid'))
    model.summary()
    # If we choose 'four', add an additional fourth layer
    if {{choice(['two', 'three'])}} == 'three':
        model.add(LSTM(n_hidden))
        model.add(Dropout({{uniform(0, 1)}}))
        

    model.add(Dense(6))
    
    model.compile(loss='categorical_crossentropy',optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},metrics=['accuracy'])
    
    model.fit(X_train,Y_train,batch_size={{choice([64, 128])}},validation_data=(X_test, Y_test),epochs=epochs)
    
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

#best_run, best_model = optim.minimize(model=model,data=data(),max_evals=10,algo=tpe.suggest,trials=Trials())


if __name__ == '__main__':

    X_train, Y_train, X_test, Y_test = data()

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
