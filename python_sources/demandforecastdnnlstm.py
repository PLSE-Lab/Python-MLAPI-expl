#!/usr/bin/env python
# coding: utf-8

# ## Summary:
# - Deep learning is used where RNN (LSTM) layers are applied to capture/learn the time sequences 
# - Multivariate and multi-step future prediction are used
#     - Multivariate to learn based on more than one feature per step
#     - multi-step future prediction to predict multi days in the future
# - Input shape is (num of samples, num of history steps = 120 days for example, input features = 7 'date details and sales')
# - Output shape is (num of samples, num of future steps = 90 days, output features = 1 'sales')
# - To construct more important features, dates are expanded to separate columns, day, month, year, day of the week, ...
# 
# This code is inspired by the tensorflow [time series tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)   
# Nice data analysis can be found [here](https://www.kaggle.com/thexyzt/keeping-it-simple-by-xyzt) 
# 
# ## Notes:
# - We have 50 items and 10 stores
# - We should have 500 models!
# - However, we can train one model for all items and stores (One global model)
# - At test time, we give the model the most recent historical data for a given item and store, and then ask the trained model to predict the next future steps
# 
# 
# *This code is for demonstration purposes only*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 20)

PATH = "../input/demand-forecasting-kernels-only"
train = pd.read_csv(f"{PATH}/train.csv", low_memory=False, parse_dates=['date'], index_col=['date'])
test = pd.read_csv(f"{PATH}/test.csv", low_memory=False, parse_dates=['date'], index_col=['date'])
sample_sub = pd.read_csv(f"{PATH}/sample_submission.csv")


# In[ ]:


display(train)
display(test)


# In[ ]:


# Expand with more date columns
def expand_df(df, istest = True):
    data = df.copy()
    data['day'] = data.index.day
    data['month'] = data.index.month
    data['year'] = data.index.year
    data['dayofweek'] = data.index.dayofweek
    if istest:
        data = data[['store', 'item', 'day', 'month', 'year', 'dayofweek', 'sales']]
    else:
        data = data[['id', 'store', 'item', 'day', 'month', 'year', 'dayofweek']]
    return data


# In[ ]:


train_data = expand_df(train)
display(train_data)

grand_avg = train_data.sales.mean()
print(f"The grand average of sales in this dataset is {grand_avg:.4f}")


# In[ ]:


test_data = expand_df(test, istest=False)
display(test_data)


# In[ ]:


agg_year_item = pd.pivot_table(train_data, index='year', columns='item', values='sales', aggfunc=np.mean).values
agg_year_store = pd.pivot_table(train_data, index='year', columns='store', values='sales', aggfunc=np.mean).values


# In[ ]:


agg_year_item.shape, agg_year_store.shape


# In[ ]:


display(pd.pivot_table(train_data, index='year', columns='store', values='sales', aggfunc=np.mean))


# In[ ]:


plt.figure(figsize=(12, 7))
plt.subplot(121)
plt.plot(agg_year_item) #/ agg_year_item.mean(0)[np.newaxis])
plt.title("Items")
plt.xlabel("Year")
plt.ylabel("Relative Sales")
plt.subplot(122)
plt.plot(agg_year_store) #/ agg_year_store.mean(0)[np.newaxis])
plt.title("Stores")
plt.xlabel("Year")
plt.ylabel("Relative Sales")
plt.show()


# In[ ]:


TRAIN_SPLIT = 40000
dataset = train_data.values
dataset = dataset[-50000:]
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)


# In[ ]:


# data normalization
dataset = (dataset-data_mean)/data_std


# In[ ]:


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)


# In[ ]:


past_history = 120 # history length
future_target = 90  # three future months as required to predict
STEP = 1 # every day

x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 6], 0, TRAIN_SPLIT,
                                                 past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 6], TRAIN_SPLIT, None,
                                             past_history,
                                             future_target, STEP)


# In[ ]:


print ('Single window of past history : {}'.format(x_train_multi[0].shape))
print ('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))


# In[ ]:


import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# In[ ]:


BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()


# In[ ]:


def create_time_steps(length):
  return list(range(-length, 0))


# In[ ]:


def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()


# In[ ]:


def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 6]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'b-',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'r-',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()


# In[ ]:


for x, y in train_data_multi.take(3):
  multi_step_plot(x[0], y[0], np.array([0]))


# In[ ]:


multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(y_train_multi.shape[-1]))

# multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')
multi_step_model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=1.0), loss='mse')
multi_step_model.summary()


# In[ ]:


# sanity
for x, y in val_data_multi.take(1):
  print (x.shape, multi_step_model.predict(x).shape)


# In[ ]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, cooldown=4, patience=3, min_lr=0.00001, verbose=1)
early_stop = tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
multi_step_history = multi_step_model.fit(train_data_multi, epochs=25,
                                          steps_per_epoch=400,
                                          validation_data=val_data_multi,
                                          validation_steps=50, callbacks=[reduce_lr, early_stop]) 


# In[ ]:


plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')


# Do not worry about the overfitting, we used **EarlyStopping** and **restore_best_weights**

# In[ ]:


for x, y in val_data_multi.take(3):
  multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])


# ## What is Next:
# - Prepare the whole data where the sequences are separated at item/store boundaries (Where no sequences containing different item/store)
# - Use one-hot-encoding for items and stores  
# - Hyper-parameter optimization for the model
# - Train, evaluate and then test the model on the required test set
# - Submit the results :)   
