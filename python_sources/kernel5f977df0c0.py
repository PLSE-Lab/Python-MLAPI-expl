#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

print(tf.__version__)


# In[ ]:


raw_dataset = pd.read_csv('./bike-rental-prediction/train.csv')


# In[ ]:


dataset = raw_dataset.copy()
dataset.tail()


# In[ ]:


# transforming dteday to day month and year
dataset['day'] = pd.DatetimeIndex(dataset['dteday']).day
dataset['month'] = pd.DatetimeIndex(dataset['dteday']).month
dataset['year'] = pd.DatetimeIndex(dataset['dteday']).year

# removing not important columns
dataset.drop(columns = ['instant','dteday', 'mnth', 'casual', 'registered'],inplace=True)
dataset.tail()


# ## Split

# In[ ]:


# splitting to testing and training datasets
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# In[ ]:


train_stats = train_dataset.describe()
train_stats.pop("cnt")
train_stats = train_stats.transpose()
train_stats


# In[ ]:


train_labels = train_dataset.pop('cnt')
test_labels = test_dataset.pop('cnt')


# ## Normalization of the dataset

# In[ ]:


# normalization of dataset
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# In[ ]:


normed_train_data.head()


# ## Model

# In[ ]:


from tensorflow.keras import backend as K

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
def rmsle(y, y0):
        return K.sqrt(K.mean(K.square(tf.math.log1p(y) - tf.math.log1p(y0))))


# In[ ]:


def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, kernel_constraint=tf.keras.constraints.NonNeg())
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss=root_mean_squared_error,
                optimizer=optimizer,
                metrics=[root_mean_squared_error, rmsle,'mae', 'mse'])

    return model


# In[ ]:


model = build_model()


# In[ ]:


model.summary()


# ## Train

# In[ ]:


EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])


# In[ ]:


# history of loss and metrics
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[ ]:


plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)


# In[ ]:


plotter.plot({'Basic': history}, metric = "loss")


# In[ ]:


loss, root_mean_squared_error, rmsle, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)


# In[ ]:


test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('CNT')
plt.ylabel('CNT Predictions')
plt.plot()


# In[ ]:


test_labels[0:10]


# In[ ]:


test_predictions[0:10]


# ## Generating of csv result

# In[ ]:


raw_test_dataset = pd.read_csv('./bike-rental-prediction/test.csv')
original_test_dataset = raw_test_dataset.copy()


# In[ ]:


# removing not important columns
original_test_dataset['day'] = pd.DatetimeIndex(original_test_dataset['dteday']).day
original_test_dataset['month'] = pd.DatetimeIndex(original_test_dataset['dteday']).month
original_test_dataset['year'] = pd.DatetimeIndex(original_test_dataset['dteday']).year

original_test_dataset_instant  = original_test_dataset.pop('instant')
original_test_dataset.drop(columns = ['dteday', 'mnth'],inplace=True)

original_test_dataset.tail()


# In[ ]:


normed_original_test_data = norm(original_test_dataset)


# In[ ]:


normed_original_test_data.head()


# In[ ]:


original_test_predictions = model.predict(normed_original_test_data).flatten()


# In[ ]:


original_test_predictions


# In[ ]:


len(original_test_predictions)


# In[ ]:


result_file = open('./results/r2.txt', 'w')
result_file.write("instant,cnt\n")

for i in range(0, len(original_test_dataset_instant) - 1):
    result_file.write(str(original_test_dataset_instant[i]) + "," + str(original_test_predictions[i]) + "\n")
    
result_file.close()


# In[ ]:





# In[ ]:





# In[ ]:




