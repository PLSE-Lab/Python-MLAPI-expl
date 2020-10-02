#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


train_df = pd.read_csv('../input/train_V2.csv')


# In[3]:


train_df.head()


# In[4]:


cols_to_keep = [
#     'Id', # Unique identifier of person
    'killPlace', # May be required
    'kills', # more kills more chances to survive
    'matchDuration', # Longer to live
    'matchType', # In which match longest lived
    'maxPlace', # Continuous Roaming in map
#     'rankPoints', # Needs to be combined with matchType for more effect
    'rideDistance', # More vehicle use, less chances to get killed
    'swimDistance', # More travel, less chances to get killed
    'walkDistance', # More travel, less chances to get killed
    'winPlacePerc', # Label
]


# In[5]:


for col in train_df.columns:
    if col not in cols_to_keep:
        train_df = train_df.drop(col, axis=1)


# In[6]:


train_df.head()


# In[7]:


train_df['matchType'].unique()


# In[8]:


dict_map = {
    'squad-fpp': 0,
    'duo': 1,
    'solo-fpp': 2,
    'squad': 3,
    'duo-fpp': 4,
    'solo': 5,
    'normal-squad-fpp': 6,
    'crashfpp': 7,
    'flaretpp': 8,
    'normal-solo-fpp': 9,
    'flarefpp': 10,
    'normal-duo-fpp': 11,
    'normal-duo': 12,
    'normal-squad': 13,
    'crashtpp': 14,
    'normal-solo': 15
}


# In[9]:


train_df['matchType'] = train_df['matchType'].map(dict_map)


# In[10]:


train_df.head()


# In[11]:


from sklearn import preprocessing
x = train_df.drop('winPlacePerc', axis=1).values
y = train_df['winPlacePerc'].values
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x)


# In[14]:


df.head()


# In[15]:


print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)


# In[16]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

dropout = False
dropoutVal = 0.2
epochs = 2
batch_size = 512


# In[17]:


def model():
    model = Sequential()
    model.add(Dense(500, input_shape=(8, ), activation='relu'))
    if dropout:
        model.add(Dropout(dropoutVal))
    model.add(Dense(300, activation='relu'))
    if dropout:
        model.add(Dropout(dropoutVal))
    model.add(Dense(50, activation='relu'))
    if dropout:
        model.add(Dropout(dropoutVal))
    model.add(Dense(1, activation='relu'))
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


# In[18]:


model =  model()


# In[19]:


history = model.fit(xTrain, yTrain, epochs=epochs, validation_data=(xTest, yTest), batch_size=batch_size)


# In[22]:


test_df = pd.read_csv('../input/test_V2.csv')


# In[23]:


test_df.columns


# In[26]:


index_df = test_df['Id']
for col in test_df.columns:
    if col not in cols_to_keep:
        test_df = test_df.drop(col, axis=1)


# In[27]:


test_df['matchType'] = test_df['matchType'].map(dict_map)


# In[28]:


test_df.head()


# In[29]:


t = test_df.values
min_max_scaler = preprocessing.MinMaxScaler()
t = min_max_scaler.fit_transform(t)


# In[30]:


pred = model.predict(t)


# In[31]:


final_sub = []
for p in pred:
    final_sub.append(p[0])

submission = pd.DataFrame({
    'Id': index_df,
    'winPlacePerc': final_sub
})


# In[ ]:


submission.to_csv('submission.csv', index=False)


# <a href='submission.csv'>Download File</a>

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




