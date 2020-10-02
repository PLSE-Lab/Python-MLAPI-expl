#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from tqdm import tqdm
warnings.filterwarnings('ignore')


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
print (train_data.shape)
train_data.head()


# In[ ]:


test_data = pd.read_csv('../input/test.csv')
print (test_data.shape)
test_data.head()


# In[ ]:


test_data['winPlacePerc'] = 0


# In[ ]:


train_data.describe()


# In[ ]:


test_data_id = test_data['Id']


# In[ ]:


train_data = train_data.drop(['Id', 'groupId', 'matchId'], 1)
test_data = test_data.drop(['Id', 'groupId', 'matchId'], 1)


# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
test_data = sc.fit_transform(test_data)
test_data = sc.transform(test_data)


# In[ ]:


y_train = train_data[['winPlacePerc']]
X_train = train_data.drop(['winPlacePerc'], 1)


# In[ ]:


X_train.shape


# In[ ]:


X_test = test_data


# In[ ]:


from keras import Sequential
from keras.layers import Dense, Dropout, Input


# In[ ]:


model = Sequential()
model.add(Dense(80,input_dim=X_train.shape[1],activation='selu'))
model.add(Dense(160,activation='selu'))
model.add(Dense(320,activation='selu'))
model.add(Dropout(0.1))
model.add(Dense(160,activation='selu'))
model.add(Dense(80,activation='selu'))
model.add(Dense(40,activation='selu'))
model.add(Dense(20,activation='selu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=70,batch_size=100000)


# In[ ]:


prediction = model.predict(X_test)


# In[ ]:


prediction = prediction.ravel()


# In[ ]:


prediction = pd.Series(prediction, name='winPlacePerc')


# In[ ]:


prediction


# In[ ]:


test_data['winPlacePerc'] = prediction


# In[ ]:


submission = test_data[['Id', 'winPlacePerc']]


# In[ ]:


submission.to_csv('pubg_submission.csv', index = False)


# In[ ]:




