#!/usr/bin/env python
# coding: utf-8

# # Starter with Neural Networks
# There is almost no work with features, I only split the datetime column into 6 columns, one-hot encoded 'passenger_count', extracted order ID from 'key' and used two features from the baseline kernel. The model is flawed and not tuned at all, its only purpose was to make sure that loss goes down no matter what, hence dropout+L2+BN. I almost purposefully made a bunch of mistakes in hope that somebody publicly corrects them.
# 
# Despite all that, I achieved 3.95 MSE with 10M samples and 3.83 MSE with all data. There is plenty of work ahead, though.

# In[ ]:


# Initial Python environment setup...
import numpy as np # linear algebra
import pandas as pd # CSV file I/O (e.g. pd.read_csv)
import os # reading the input files we have access to

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from keras_tqdm import TQDMNotebookCallback


# In[ ]:


#features from basic linear model kernel
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()


# # Loading and preprocessing data in its entirety
# I managed to load and preprocess the whole dataset with pandas, but it took ~20 minutes. Again, I'm uploading it so that somebody shows how to do it correctly with, I dunno, Dask. 

# In[ ]:


filename = 'train.csv'
dfs = []
chunksize = 10 ** 6
for chunk in tqdm(pd.read_csv(filename, chunksize=chunksize)):
    #preprocessing section
    add_travel_vector_features(chunk)
    chunk = chunk.dropna(how = 'any', axis = 'rows')
    chunk = chunk[(chunk.abs_diff_longitude < 5.0) & (chunk.abs_diff_latitude < 5.0)]
    chunk = chunk[(chunk.passenger_count > 0) & (chunk.passenger_count <= 6)]
    chunk[['date','time','timezone']] = chunk['pickup_datetime'].str.split(expand=True)
    chunk[['year','month','day']] = chunk['date'].str.split('-',expand=True).astype('int64')
    chunk[['hour','minute','second']] = chunk['time'].str.split(':',expand=True).astype('int64')
    chunk['year_after_0'] = chunk['year'] - np.min(chunk['year'])
    chunk[['trash', 'order_no']] = chunk['key'].str.split('.',expand=True)
    chunk['order_no'] = chunk['order_no'].astype('int64')
    chunk = pd.concat([chunk,pd.get_dummies(chunk['passenger_count'],prefix='pass')], axis =1)
    chunk = chunk.drop(['timezone','date','time', 'pickup_datetime','trash','key','passenger_count'], axis = 1)
    #append chunk to the list
    dfs.append(chunk)


# In[ ]:


get_ipython().run_cell_magic('time', '', '#concatenate all chunk in one big-ass DataFrame\ntrain_df = pd.concat(dfs)')


# In[ ]:


#delete the chunks as I only have 16 GB RAM
del dfs


# In[ ]:


train_df.head()


# In[ ]:


train_df.shape


# In[ ]:


X_train = train_df.drop(['fare_amount'],axis=1)
Y_train = train_df['fare_amount']


# In[ ]:


del train_df


# In[ ]:


scaler = StandardScaler()
y_scaler = StandardScaler()


# In[ ]:


#scale the data so that columns have zero mean and unit variance
train = scaler.fit_transform(X_train.values)
y_train =  y_scaler.fit_transform(Y_train.values.reshape(-1,1))


# In[ ]:


del X_train
del Y_train


# In[ ]:


import keras
import tensorflow as tf


# In[ ]:


#some imports are unnecessary
from keras import layers
from keras.layers import Input, Dropout,Dense, Activation, BatchNormalization
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint,  ReduceLROnPlateau
from keras.regularizers import l2
from keras.optimizers import Adam


# # Model

# In[ ]:


model = keras.Sequential([
    keras.layers.Dense(1024,kernel_initializer = glorot_uniform(),
              kernel_regularizer = l2(1e-2)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(tf.nn.leaky_relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024,kernel_initializer = glorot_uniform(),
              kernel_regularizer = l2(1e-2)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(tf.nn.leaky_relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024,kernel_initializer = glorot_uniform(),
              kernel_regularizer = l2(1e-2)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(tf.nn.leaky_relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1024,kernel_initializer = glorot_uniform(),
              kernel_regularizer = l2(1e-2)),
    keras.layers.BatchNormalization(),
    keras.layers.Activation(tf.nn.leaky_relu),
    keras.layers.Dense(1, activation=tf.nn.leaky_relu)
])


# In[ ]:


model.compile(optimizer=Adam(5e-4), 
              loss='mean_squared_error')


# # Callbacks

# In[ ]:


filepath = './model_weights/weights-improvement-55M-{epoch:02d}-{val_loss:.4f}.hdf5'
best_callback = ModelCheckpoint(filepath, 
                                save_best_only=True)
lr_sched = ReduceLROnPlateau(monitor='val_loss', factor = 0.2, patience = 5, verbose = 1)
tqdm_callback = TQDMNotebookCallback(leave_inner=True,metric_format="{name}: {value:0.5f}")


# # Training

# In[ ]:


history = model.fit(train, y_train, 
          epochs=20,
          verbose=0,
          batch_size=2048,
          validation_split=0.0002,
          callbacks=[tqdm_callback,best_callback, lr_sched])


# # Load best result

# In[ ]:


model.load_weights('./model_weights/weights-improvement-55M-19-0.0471.hdf5')


# # Load and preprocess test data

# In[ ]:


test_df = pd.read_csv('test.csv')
test_df.dtypes


# In[ ]:


key = test_df.key
add_travel_vector_features(test_df)
test_df[['date','time','timezone']] = test_df['pickup_datetime'].str.split(expand=True)
test_df[['year','month','day']] = test_df['date'].str.split('-',expand=True).astype('int64')
test_df[['hour','minute','second']] = test_df['time'].str.split(':',expand=True).astype('int64')
test_df['year_after_0'] = test_df['year'] - np.min(test_df['year'])
test_df[['trash', 'order_no']] = test_df['key'].str.split('.',expand=True)
test_df['order_no'] = test_df['order_no'].astype('int64')
test_df = pd.concat([test_df,pd.get_dummies(test_df['passenger_count'],prefix='pass')], axis =1)
test_df = test_df.drop(['timezone','date','time', 'pickup_datetime','trash','key','passenger_count'], axis = 1)
# Predict fare_amount on the test set using our model (w) tested on the testing set.
test_df.shape


# # Inference and submission

# In[ ]:


test = scaler.transform(test_df.values)
y_test = model.predict(test)
y_test = y_scaler.inverse_transform(y_test).reshape(-1)
# Write the predictions to a CSV file which we can submit to the competition.
submission = pd.DataFrame(
    {'key': key, 'fare_amount': y_test},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission_100.csv', index = False)

print(os.listdir('.'))


# # What's next
# 1. Extract better features.
# 2. Choose a better architecture.
# 3. Tune the hyperparameters.
# 4. Forget all that and resort to XGBoost and ensembling.

# In[ ]:




