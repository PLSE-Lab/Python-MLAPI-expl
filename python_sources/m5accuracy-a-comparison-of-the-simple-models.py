#!/usr/bin/env python
# coding: utf-8

# This notebook is created by Bang Nguyen for beginner of the M5 competition to easily understand and start making your first notebook to solve the problem.
# 
# I started to explorer the M5 dataset here
# 
# https://www.kaggle.com/nlebang/m5-forecasting-data-explanation
# 
# Starting from the average 30 days simplest solution. We got the base-line score of 1.07118
# 
# In this notebooks, I compares the results of other simple solutions together.
# 
# I do collect some codes from other notebooks.
# 
# Give me an UPVOTE, if it is useful!
# 
# Thank you!

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


# From this below notebook, we have the simplest solution (1.07118) by averaging 30 latest days of sales and use that for all predicted 28 days.
# 
# https://www.kaggle.com/robikscube/m5-forecasting-starter-data-exploration 
# 
# I digged all the notebooks of M5 compitetion and select other solutions
# 
# # 1. THE SIMPLEST LSTM MODEL
# 
# 1. https://www.kaggle.com/graymant/baseline-lstm-example -> This notebook having 2.07228 score employed the simple long-short-term-memory (LSTM) model.

# In[ ]:


import gc

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import os

from tqdm import trange, tqdm_notebook

from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM 
from keras.layers import Conv1D
from keras.utils import to_categorical
from keras.layers import MaxPooling1D
from keras.layers import  GlobalAveragePooling1D
from keras.utils import to_categorical

import tensorflow as tf


# In[ ]:


input_path = "../input/m5-forecasting-accuracy"

def get_salesval_coltypes():
    keys = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'] +         [f"d_{i}" for i in range(1, 1914)]
    values = ['object', 'category', 'category', 'category', 'category', 'category'] +        ["uint16" for i in range(1, 1914)]
    return dict(zip(keys, values))

submission = pd.read_csv(os.path.join(input_path, 'sample_submission.csv'))
sales_train_val = pd.read_csv(os.path.join(input_path, 'sales_train_validation.csv'), 
                              dtype=get_salesval_coltypes())

#calendar = pd.read_csv(os.path.join(input_path, 'calendar.csv'))
#sell_prices = pd.read_csv(os.path.join(input_path, 'sell_prices.csv'))


# In[ ]:


# Take the timeseries data is in columns d_1 to d_1913 in "sales_train_val"
data = sales_train_val.iloc[:, 6:]
data.head()


# In[ ]:


#Normalize the data
#Or we can perform this step later using MinMaxScaler() or StandardScaler()
#data = (data-data.min())/(data.max()-data.min())
#data.head()


# In[ ]:


# For later - test train split, for now just get shapes right
#Now lets reshape the data into the 3D inputs required by the LSTM. 
#As a starting point we'll use input sequences of 100 timesteps to predict 28 steps ahead.
# For LSTM, X needs to be a stack of shape (samples, timesteps, features)
# So aiming at a shape of  = (~order of 30490 * timesteps, 28, 1)
base = []
predictions = []

timesteps = 100
prediction_steps = 28

# Well just iterate through slicing timesteps until we get somewhat near the end. With a
# proper train test split, we could be more precise
for i in range(1, 12):
    #Take out the data in the period that we need
    samples = data.iloc[:, i*timesteps:i*timesteps+timesteps]
    preds = data.iloc[:, i*timesteps+timesteps:i*timesteps+timesteps+prediction_steps]
    #Add to a new data set
    base.extend(samples.to_numpy())
    predictions.extend(preds.to_numpy())
    print(f"Samples {samples.shape}, preds {preds.shape}")


# In[ ]:


#Normalize or Standarlize the data
input_scaler = MinMaxScaler()
output_scaler = StandardScaler()

# Scale and reshape our input
X_train = np.array(base)
input_scaler.fit(X_train)
X_train = input_scaler.transform(X_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Scale our prediction labels
Y_train_orig = np.array(predictions)
output_scaler.fit(Y_train_orig)
Y_train = output_scaler.transform(Y_train_orig)
print(X_train.shape)
print(Y_train.shape)

# Note this could be horrible on memory. Later, need to look at generating this in batches
del predictions
del base
gc.collect()


# In[ ]:


#Lets create the required LSTM model based on the input shapes
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

steps = X_train.shape[1]
n_features = X_train.shape[2]
n_steps_out = Y_train.shape[1]

model = tf.keras.Sequential()
model.add(CuDNNLSTM(100, return_sequences=True, input_shape=(steps, n_features)))
model.add(CuDNNLSTM(50))
#model.add(LSTM(100, activation='relu'))
model.add(tf.keras.layers.Dense(n_steps_out))
# this loss needs changing to competition loss.
model.compile(optimizer='adam', loss=root_mean_squared_error) 


# In[ ]:


#Train the model
#%%time

# 0.6345 200 56
# 0.5633 200 56 4m 14s

model.fit(X_train, Y_train, epochs=100, verbose=1)


# In[ ]:


#Now we have a trained model, we need to take the last set of timesteps 
#from the input data and get our final predictions.
# Take a slice of n{timesteps} from the input data
x_pred = data.iloc[:,-timesteps:].to_numpy()

# Reshape to fit the format for input scalar
x_pred = x_pred.reshape((len(sales_train_val), x_pred.shape[1]))
# Normalize the input
x_pred = input_scaler.transform(x_pred)
# Reshape to fit the format for LSTM model
x_pred = x_pred.reshape((len(sales_train_val), x_pred.shape[1], 1))

# Get our predictions
raw_pred = model.predict(x_pred)

# Inverse to transform to get the predictions at the right scale
all_pred = output_scaler.inverse_transform(raw_pred)
# Round the predictions back to integers
all_pred = np.round(np.abs(all_pred))


# To finish, we just need to stack our predictions into the format required for the submission file.
# 
# As we only predicted one set of 28 days, lets just stack them twice into the results file. This wouldn't be satisfactory for a final attempt on the private leaderboard, but for now while developing a model, it will do.

# In[ ]:


# Stack our predictions into a dataframe
validation = pd.concat([pd.DataFrame(all_pred[:,0:prediction_steps]), pd.DataFrame(all_pred[:,-prediction_steps:])])
validation = validation.astype(int)

# Reset index to match the submission dataframe
validation.reset_index(inplace=True, drop=True)

# Add the id column from the submission dataframe to our results
validation['id'] = submission.id
validation = validation.reindex(
        columns=['id'] + [c for c in validation.columns if c != 'id'], copy=False)

# Add the correct colummn names for the submission file format
validation.columns = ['id'] + [f"F{i}" for i in range(1, 29)]

validation.to_csv('submission_simplestLSTM.csv', index=False)

