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

import keras
from keras import models
from keras import layers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
y = np.log(train.SalePrice)
X = train.drop(['SalePrice'], axis=1)
print(X.shape, test.shape)


for i in X.columns:
    if X[i].dtype == 'object':
        X[i] = X[i].fillna('0')
    else:
        X[i] = X[i].fillna(-1)
        
for i in test.columns:
    if test[i].dtype == 'object':
        test[i] = test[i].fillna('0')
    else:
        test[i] = test[i].fillna(-1)
        
X = pd.get_dummies(X)
test = pd.get_dummies(test)
print(X.shape)
print(test.shape)

cols = [value for value in test.columns if value in X.columns]
X = X[cols]
test = test[cols]

train_data, validation_data, train_targets, validation_targets = train_test_split(X, y,test_size=0.2)
print(train_data.shape)
print(validation_data.shape)
print(test.shape)

# Centering and scaling data
mean= train_data.mean(axis = 0)
train_data -= mean
std = train_data.std(axis = 0)
train_data /= std

validation_data -= mean
validation_data /= std

test -= mean
test /= std

train_data = train_data.fillna(0)
validation_data = validation_data.fillna(0)
test = test.fillna(0)

train_data.head()

# Model Build

def build_model_dropout():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    model.summary()
    return model
    
# K - Fold cross validation

k=2
num_val_samples = len(train_data) //k
all_scores = []

num_epochs = 5000
all_mae_histories = []
for i in range(k):
    print('processing fold #',i)
#     val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
#     val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]
    val_data = validation_data.copy()
    val_targets = validation_targets.copy()
    
    partial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)
    
    model = build_model_dropout()
    
    # checkpoint
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_learning_rate = ReduceLROnPlateau(monitor = "val_loss", factor = 0.9,patience=100, min_lr=0.001,  verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300)
    callbacks_list = [checkpoint,reduce_learning_rate,early_stopping]
    
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data,val_targets), epochs=num_epochs, 
                        callbacks=callbacks_list, batch_size=16,verbose=2)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

# estimate accuracy on whole dataset using loaded weights
scores = model.evaluate(val_data, val_targets, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))

#Submission
predictions = model.predict(test)
submission = pd.read_csv('../input/sample_submission.csv')
submission.SalePrice = np.exp(predictions)
submission.to_csv('submission.csv', index=False)