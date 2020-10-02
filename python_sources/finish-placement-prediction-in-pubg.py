#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train_data = pd.read_csv('../input/train_V2.csv')


# In[ ]:


train_data.head()


# In[ ]:


train_data.describe()


# In[ ]:


train_data.shape


# In[ ]:


train_data = train_data.dropna()


# In[ ]:


train_data.isna().sum()


# In[ ]:


train_data.info()


# In[ ]:


_ = plt.figure(figsize=(30, 20))
p = sns.heatmap(train_data.corr(), annot=True)


# The plot above gives the correlation between the different features

# This plot can give us a lot of information about the data we want to predict. We can see from the plot that the values which highly effect the *winPlacePerc* are *walkDistance*, *weaponsAquired*, *killPlace*, *boosts*. *killPlace*  have highly negative effect on the win percentage whereas others have a significantly positive effect. 

# In[ ]:


Y_train = train_data['winPlacePerc']
X_train = train_data.drop(columns=['Id', 'groupId', 'matchId', 'winPlacePerc', 'matchType', 'DBNOs', 'headshotKills', 'matchDuration', 'maxPlace', 'numGroups', 'roadKills', 'vehicleDestroys', 'swimDistance'])

Y = Y_train.values
X = X_train.values


# In[ ]:


import tensorflow as tf
from tensorflow import keras
from keras import models
from keras import layers
from keras import Sequential
from keras.layers import Dense, Dropout, Input


# In[ ]:


model = Sequential()
model.add(Dense(80,input_dim=X_train.shape[1],activation='relu'))
model.add(Dense(160,activation='relu'))
model.add(Dense(320,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(160,activation='relu'))
model.add(Dense(80,activation='relu'))
model.add(Dense(40,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# In[ ]:


model.summary()


# In[ ]:


history = model.fit(X, Y, epochs=50,
        batch_size=10000,
        validation_split=0.2,
        verbose=2)


# In[ ]:


plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.legend(['mean_absolute_error', 'val_mean_absolute_error'])


# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])


# In[ ]:


test_data = pd.read_csv('../input/test_V2.csv')


# In[ ]:


X_test = test_data.drop(columns=['Id', 'groupId', 'matchId', 'matchType', 'DBNOs', 'headshotKills', 'matchDuration', 'maxPlace', 'numGroups', 'roadKills', 'vehicleDestroys', 'swimDistance'])


# In[ ]:


predictions = model.predict(X_test).ravel()


# In[ ]:


predictions


# In[ ]:


sample_sub = pd.read_csv('../input/sample_submission_V2.csv')


# In[ ]:


sample_sub["winPlacePerc"] = predictions
sample_sub.head()


# In[ ]:


sample_sub.to_csv('sample_submission_v1.csv', index=False)


# In[ ]:




