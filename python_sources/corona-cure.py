#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


master = pd.read_csv('/kaggle/input/coronavirus-covid19-drug-discovery/master_results_table.csv')
sars_med = pd.read_csv('/kaggle/input/coronavirus-covid19-drug-discovery/sars_results.csv')
mers_med = pd.read_csv('/kaggle/input/coronavirus-covid19-drug-discovery/mers_results.csv')


# In[ ]:


master


# In[ ]:


sars_med


# In[ ]:


mers_med


# In[ ]:


train_X = sars_med.merge(mers_med, on='Unnamed: 0')


# In[ ]:


train_X


# In[ ]:


train_X['Unnamed: 0']


# In[ ]:


train_y = train_X[]


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
#create model
model = Sequential()

#get number of columns in training data
n_cols = train_X.shape[1]

#add model layers
model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))


# In[ ]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[ ]:


from keras.callbacks import EarlyStopping
#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)
#train model
model.fit(train_X, train_y, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])
test_y_predictions = model.predict(test_X)


# In[ ]:


test_y_predictions = model.predict(test_X)


# In[ ]:


#create model
model_mc = Sequential()

#add model layers
model_mc.add(Dense(200, activation='relu', input_shape=(n_cols,)))
model_mc.add(Dense(200, activation='relu'))
model_mc.add(Dense(200, activation='relu'))
model_mc.add(Dense(1))

#compile model using mse as a measure of model performance
model_mc.compile(optimizer='adam', loss='mean_squared_error')
#train model
model_mc.fit(train_X, train_y, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])

