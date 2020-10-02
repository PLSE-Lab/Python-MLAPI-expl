# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Any results you write to the current directory are saved as output.

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

train = pd.read_csv('../input/carbon_nanotubes.csv')

# simple data cleaning
train.replace(',','.', inplace = True, regex=True)
train = train.apply(pd.to_numeric) 

# seperate the features and targets
X_set = train[['Chiral indice n','Chiral indice m','Initial atomic coordinate u', 'Initial atomic coordinate v', 'Initial atomic coordinate w']]
y_set = train[["Calculated atomic coordinates u'", "Calculated atomic coordinates v'", "Calculated atomic coordinates w'"]]

# train set 80% and test set 20%
X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=0.2)

# scale the dataset because of two features n and m
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

# Feedforward neural network model structure: one hidden layer with 20 neurons under learning rate 0.02
model = Sequential()
model.add(Dense(20, input_shape=(5,), activation = 'softmax'))
model.add(Dense(3,))
model.compile(Adam(lr=0.02), 'mean_squared_error')
history = model.fit(X_train, y_train, epochs = 1000, validation_split = 0.2,verbose = 0)

# Plots 'history'
history_dict=history.history
loss_values = history_dict['loss']
val_loss_values=history_dict['val_loss']
plt.plot(loss_values,'bo',label='training loss')
plt.plot(val_loss_values,'r',label='training loss val')
plt.show()

# R square score to check the model accurary
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_test_pred)))