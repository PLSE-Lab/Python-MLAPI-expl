#!/usr/bin/env python
# coding: utf-8

# This notebook should serve as a very simple example and starting point of how to shape the data to be used in an LSTM model. Note that for now, we ignore all features except the timeseries data and choose a somewhat arbitrary number of timesteps. It should however give a basic starting point and an indicator of a baseline score for using an LSTM model.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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


# First lets load and compress the data

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


# Now lets reshape the data into the 3D inputs required by the LSTM. As a starting point we'll use input sequences of 100 timesteps to predict 28 steps ahead.

# In[ ]:


# Prepare scalars to normalize data
input_scaler = MinMaxScaler()
output_scaler = StandardScaler()

# Our timeseries data is in cols d_1 to d_1913
data = sales_train_val.iloc[:, 6:]
#data = (data-data.min())/(data.max()-data.min())

# For LSTM, X needs to be a stack of shape (samples, timesteps, features)
# So aiming at a shape of  = (~order of 30490 * timesteps, 28, 1)

# For later - test train split, for now just get shapes right
base = []
predictions = []

timesteps = 100
prediction_steps = 28

# Well just iterate through slicing timesteps until we get somewhat near the end. With a
# proper train test split, we could be more precise
for i in range(1, 12):
    samples = data.iloc[:, i*timesteps:i*timesteps+timesteps]
    preds = data.iloc[:, i*timesteps+timesteps:i*timesteps+timesteps+prediction_steps]
    base.extend(samples.to_numpy())
    predictions.extend(preds.to_numpy())
    #print(f"Samples {samples.shape}, preds {preds.shape}")

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


# Now we've prepared the data, lets create the required LSTM model based on the input shapes

# In[ ]:


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
model.compile(optimizer='adam', loss=root_mean_squared_error) # this loss needs changing to competition loss.


# And finally, train the model

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# 0.6345 200 56\n# 0.5633 200 56 4m 14s\n\nmodel.fit(X_train, Y_train, epochs=2, verbose=1)')


# Now we have a trained model, we need to take the last set of timesteps from the input data and get our final predictions.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Take a slice of n{timesteps} from the input data\nx_pred = data.iloc[:,-timesteps:].to_numpy()\n\n# Reshape to fit the format for input scalar\nx_pred = x_pred.reshape((len(sales_train_val), x_pred.shape[1]))\n# Normalize the input\nx_pred = input_scaler.transform(x_pred)\n# Reshape to fit the format for LSTM model\nx_pred = x_pred.reshape((len(sales_train_val), x_pred.shape[1], 1))\n\n# Get our predictions\nraw_pred = model.predict(x_pred)\n\n# Inverse to transform to get the predictions at the right scale\nall_pred = output_scaler.inverse_transform(raw_pred)\n# Round the predictions back to integers\nall_pred = np.round(np.abs(all_pred))')


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

validation.to_csv('submission.csv', index=False)

