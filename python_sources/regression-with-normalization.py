#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()


# In[ ]:


concrete_data.shape


# In[ ]:


concrete_data.describe()


# In[ ]:


concrete_data.isnull().sum()


# In[ ]:


concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column


# In[ ]:


predictors.head()


# In[ ]:


target.head()


# In[ ]:


predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()


# In[ ]:


n_cols = predictors_norm.shape[1] # number of predictors


# In[ ]:


import keras


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense


# In[ ]:


# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# In[ ]:


model=regression_model()


# In[ ]:


model.fit(predictors_norm, target, validation_split=0.4, epochs=10, verbose=2)


# In[ ]:




