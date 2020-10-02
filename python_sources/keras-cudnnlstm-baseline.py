#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import os

from tqdm import trange, tqdm_notebook

from tensorflow.keras import backend as K
import tensorflow as tf


# In[ ]:


def get_salesval_coltypes():
    keys = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'] +         [f"d_{i}" for i in range(1, 1914)]
    values = ['object', 'category', 'category', 'category', 'category', 'category'] +        ["uint16" for i in range(1, 1914)]
    return dict(zip(keys, values))


# In[ ]:


timesteps = 100
y_steps = 28
input_scaler = MinMaxScaler()
output_scaler = StandardScaler()


# In[ ]:


def get_data():
    data = pd.read_csv(os.path.join("/kaggle/input/m5-forecasting-accuracy", 'sales_train_validation.csv'), dtype=get_salesval_coltypes())
    

    # Our timeseries data is in cols d_1 to d_1913
    data = data.iloc[:, 6:]
    #data = (data-data.min())/(data.max()-data.min())

    # For later - test train split, for now just get shapes right
    x = []
    y = []

    # Well just iterate through slicing timesteps until we get somewhat near the end. With a
    # proper train test split, we could be more precise
    for i in range(1, 12):
        x_data = data.iloc[:, i*timesteps:i*timesteps+timesteps]
        y_data = data.iloc[:, i*timesteps+timesteps:i*timesteps+timesteps+y_steps]
        x.extend(x_data.to_numpy())
        y.extend(y_data.to_numpy())
        #print(f"Samples {samples.shape}, preds {preds.shape}")

    # Scale and reshape our input
    x_train = np.array(x)
    input_scaler.fit(x_train)
    x_train = input_scaler.transform(x_train)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    # Scale our prediction labels
    y_train = np.array(y)
    output_scaler.fit(y_train)
    y_train = output_scaler.transform(y_train)
    
    # Take a slice of n{timesteps} from the input data
    x_test = data.iloc[:,-timesteps:].to_numpy()

    # Reshape to fit the format for input scalar
    x_test = x_test.reshape((len(data), x_test.shape[1]))
    # Normalize the input
    x_test = input_scaler.transform(x_test)
    # Reshape to fit the format for LSTM model
    x_test = x_test.reshape((len(data), x_test.shape[1], 1))
    return x_train,y_train,x_test

x_train,y_train,x_test = get_data()    
print(x_train.shape)
print(y_train.shape)


# In[ ]:


def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
def build_model(epochs,batch_size):
    steps = x_train.shape[1]
    n_features = x_train.shape[2]
    n_steps_out = y_train.shape[1]

    model = tf.keras.models.Sequential()
    model.add(tf.compat.v1.keras.layers.CuDNNLSTM(100, return_sequences=True, input_shape=(steps, n_features)))
    model.add(tf.compat.v1.keras.layers.CuDNNLSTM(50))
    model.add(tf.keras.layers.Dense(n_steps_out))
    model.compile(optimizer='adam', loss=root_mean_squared_error) # this loss needs changing to competition loss.
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model
epochs = 80
batch_size = 1000
model = build_model(epochs,batch_size)


# In[ ]:


def make_prediction(x_test):
    # Get our predictions
    pred = model.predict(x_test)
    submission = pd.read_csv(os.path.join("/kaggle/input/m5-forecasting-accuracy",'sample_submission.csv'))
    # Inverse to transform to get the predictions at the right scale
    pred = output_scaler.inverse_transform(pred)
    # Round the predictions back to integers
    pred = np.round(np.abs(pred))
    validation = pd.concat([pd.DataFrame(pred[:,0:y_steps]), pd.DataFrame(pred[:,-y_steps:])])
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
make_prediction(x_test)

