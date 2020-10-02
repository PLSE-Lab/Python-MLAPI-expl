#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Input, GRU, TimeDistributed, Dense
from keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse

from matplotlib import pyplot as plt


# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# # Preprocess the data

# In[3]:


sc = StandardScaler()


# In[4]:


feat_cols = [f'sensor#{i}' for i in range(12)] + ['timestamp', 'dow', 'hour_sin', 'hour_cos']


# In[5]:


for data in [train, test]:
    data['time_converted'] = pd.to_datetime(data['timestamp'], unit='ms')
    data['dow'] = data['time_converted'].apply(lambda x: x.dayofweek)
    data['hour_sin'] = data['time_converted'].apply(lambda d: np.sin(2 * np.pi * (d.hour + d.minute / 60) / 24))
    data['hour_cos'] = data['time_converted'].apply(lambda d: np.cos(2 * np.pi * (d.hour + d.minute / 60) / 24))
    for col in feat_cols:
        data[col] = sc.fit_transform(data[col].values.astype('float').reshape(-1, 1))


# In[6]:


train_x = train[feat_cols].values
train_y = train['oil_extraction'].values
test_x = test[feat_cols].values


# In[7]:


train_y = sc.fit_transform(train_y.reshape(-1, 1))


# # Split the data

# In[8]:


val_size = 8640


# In[9]:


val_x = train_x[-val_size:, :]
train_x = train_x[:-val_size, :]
val_y = train_y[-val_size:, :]
train_y = train_y[:-val_size, :]


# In[10]:


train_x.shape, train_y.shape, val_x.shape, val_y.shape, test_x.shape


# # Create the model

# In[11]:


inputs = Input(shape=(None, 16))
inter = GRU(32, return_sequences=True)(inputs)
inter = GRU(16, return_sequences=True)(inter)
outputs = TimeDistributed(Dense(1))(inter)


# In[12]:


model = Model(inputs = inputs, outputs = outputs)


# In[13]:


model.summary()


# # Train the model

# In[14]:


import random

class BatchGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size, hist_len):
        self.x = x
        self.y = y
        self.n_samples = x.shape[0]
        self.n_features = x.shape[1]
        self.batch_size = batch_size
        self.hist_len = hist_len
    
    def __len__(self):
        return (self.n_samples - self.hist_len) // self.batch_size
    
    def __getitem__(self, index):
        batch_x, batch_y = [], []
        for i in range(self.batch_size):
            start = random.randint(0, self.n_samples - self.hist_len)
            batch_x.append(self.x[start:(start+self.hist_len)].reshape(1, -1, self.n_features))
            batch_y.append(self.y[start:(start+self.hist_len)].reshape(1, -1, 1))
        return np.vstack(batch_x), np.vstack(batch_y)


# In[15]:


model.compile(Adam(0.01), 'mse')


# In[16]:


bg_train = BatchGenerator(train_x, train_y, 64, 256)
bg_val = BatchGenerator(val_x, val_y, 64, 256)


# In[17]:


model.fit_generator(bg_train, steps_per_epoch = 20, epochs = 20, validation_data = bg_val, validation_steps = 1, verbose = 1)


# In[18]:


model.compile(Adam(0.001), 'mse')


# In[19]:


bg_train = BatchGenerator(train_x, train_y, 64, 1024)
bg_val = BatchGenerator(val_x, val_y, 64, 1024)


# In[ ]:


model.fit_generator(bg_train, steps_per_epoch = 10, epochs = 10, validation_data = bg_val, validation_steps = 1, verbose = 1)


# # Evaluate the model

# In[ ]:


val_pred = model.predict(val_x.reshape(1, -1, 16)).reshape(-1, 1)


# In[ ]:


fig, ax = plt.subplots(figsize = (20, 5))
ax.plot(val_y[-200:], c = 'b')
ax.plot(val_pred[-200:], c = 'g')


# In[ ]:


val_pred_scaled = sc.inverse_transform(val_pred)
val_y_scaled = sc.inverse_transform(val_y)


# In[ ]:


mse(val_y_scaled, val_pred_scaled)


# # Save the predictions

# In[ ]:


test_pred = model.predict(test_x.reshape(1, -1, 16)).reshape(-1, 1)
test_pred_scaled = sc.inverse_transform(test_pred)


# In[ ]:


result_df = pd.DataFrame({'Expected': test_pred_scaled.reshape(-1, ), 'Id': range(test_pred.shape[0])})


# In[ ]:


result_df.to_csv('submission.csv', index = False)

