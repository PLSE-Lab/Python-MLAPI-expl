# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

train = pd.read_csv(r"/kaggle/input/train-test-universe/Glove_train_universe.csv")
test = pd.read_csv(r"/kaggle/input/train-test-universe/Glove_test_universe.csv")

train.head()

y = np.log(train['price'].values+1)
X = np.array(train.drop(['price','train_id'], axis=1))

train_data, validation_data, train_targets, validation_targets = train_test_split(X, y,test_size=0.2)
print(train_data.shape)
print(validation_data.shape)
print(test.shape)

test_data = test.drop(['train_id'], axis=1)

# Centering and scaling data
mean= train_data.mean(axis = 0)
train_data -= mean
std = train_data.std(axis = 0)
train_data /= std

validation_data -= mean
validation_data /= std

test_data -= mean
test_data /= std

# Model Build
from keras import backend as K
import keras
from keras import models
from keras import layers
from keras.layers import LeakyReLU
from keras.layers import ELU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred-y_true)))
    
def build_model_dropout():
    model = models.Sequential()
    model.add(layers.Dense(150,activation='relu', 
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(135,activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(120,activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(100,activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(80,activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(60,activation='relu'))
    model.add(layers.Dropout(0.05))
    model.add(layers.Dense(40,activation='relu'))
    model.add(layers.Dropout(0.02))
    model.add(layers.Dense(30,activation='tanh'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss=root_mean_squared_error, metrics=['mse'])
    model.summary()
    return model

num_epochs = 40
all_mae_histories = []

# Train on full data
val_data = validation_data.copy()
val_targets = validation_targets.copy()

partial_train_data = train_data.copy()
partial_train_targets = train_targets.copy()

model = build_model_dropout()
    
# checkpoint
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
reduce_learning_rate = ReduceLROnPlateau(monitor = "val_loss", factor = 0.9,patience=5, min_lr=0.001,  verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
callbacks_list = [checkpoint,reduce_learning_rate,early_stopping]

history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data,val_targets), epochs=num_epochs, 
                    callbacks=callbacks_list, batch_size=16,verbose=2)
mae_history = history.history['val_mean_squared_error']
all_mae_histories.append(mae_history)

# load the model
from keras.models import Sequential, load_model
new_model = load_model("weights.best.hdf5", custom_objects={'root_mean_squared_error': root_mean_squared_error})

# estimate accuracy on whole dataset using loaded weights
scores = new_model.evaluate(val_data, val_targets, verbose=0)
print("%s: %.4f%%" % (new_model.metrics_names[1], scores[1]))
print("%s: %.4f%%" % (new_model.metrics_names[0], scores[0]))

predictions = new_model.predict(test_data)
submission = pd.read_csv('/kaggle/input/guess-my-price/sample_submission.csv')
submission.price = np.exp(predictions)
submission.to_csv('submission.csv', index=False)