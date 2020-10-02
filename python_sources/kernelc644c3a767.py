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
import seaborn as sbn
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


data_train = pd.read_csv('../input/train_V2.csv')
data_test = pd.read_csv('../input/test_V2.csv')


# In[3]:


data_train.head()


# In[4]:


data_test.head()


# In[ ]:





# > #Feature Extraction and Feature Analysis

# Coloumn Info for test and train data set

# In[5]:


data_train.info()


# In[6]:


data_test.info()


# In[7]:


data_train.isna().sum()


# In[8]:


data_test.isna().sum(
)


# Since the training data consist of 1 null value row we cn remove the data

# In[9]:


data_train.shape


# In[10]:


data_test.shape


# In[11]:


data_train.dropna(inplace=True)


# Now both the train nd test data are null value free hence we can proceed forward
# Now,
# The data set consist of around 47, lakhs instances and various instance has same match id, that is they are in the smae match hence we can introduce a new feature, which consist of number of player in a given match

# In[12]:


data_train.groupby('matchId')[('matchId')].count()


# In[13]:


data_test.groupby('matchId')['matchId'].transform('count')


# In[14]:


data_train['Match Played'] = data_train.groupby('matchId')['matchId'].transform('count')
data_test['Match Played'] = data_test.groupby('matchId')['matchId'].transform('count')


# moving on lets represent the data using graphical represent and understand the data in both train and test set

# In[15]:


plt.subplots(figsize =(10,10))
sbn.countplot(data_train['Match Played'])


# In[16]:


data_train = pd.get_dummies(data_train,columns=['matchType'])


# In[17]:


data_train.shape


# In[18]:


data_test  = pd.get_dummies(data_test,columns= ['matchType'])


# In[19]:



data_test.shape


# In[20]:


data_train['totalDistance'] = data_train['rideDistance']+data_train['swimDistance'] + data_train['walkDistance']


# In[21]:


data_test['totalDistance'] = data_test['rideDistance']+data_test['swimDistance'] + data_test['walkDistance']


# applying random forest regressor to learn the important feature but we need to drop Id, PlayerID , Match Id since they are of object type and we already extracted the information 

# In[22]:


data_train.drop(labels=['Id','groupId','matchId'],inplace = True,axis=1)
test_id=data_test['Id']
data_test.drop(labels=['Id','groupId','matchId'],inplace = True,axis=1)


# In[23]:


data_train.shape


# In[24]:


rdata=data_train.sample(3500000)


# In[25]:


y = rdata['winPlacePerc']
X_data = rdata.drop(labels='winPlacePerc',axis=1)


# In[26]:


X_train,X_test,y_train,y_test = train_test_split(X_data,y,test_size=.3)


# In[27]:


rfr = RandomForestRegressor(n_estimators=35, min_samples_leaf=3, max_features='sqrt',
                          n_jobs=-1)


# In[28]:


rfr.fit(X_train,y_train)


# In[29]:


y_pred = rfr.predict(X_test)


# In[30]:


mean_squared_error(y_pred,y_test)


# In[31]:


mean_absolute_error(y_pred,y_test)


# In[32]:


new_data =pd.DataFrame(sorted(zip(rfr.feature_importances_, X_data.columns)),columns=['Value','Feature'])


# In[33]:


new_data # holdiing the column name and its importance value


# In[34]:


new_data = new_data.sort_values(by='Value',ascending=False)[:25]


# In[35]:


new_data.shape


# In[36]:



cols=new_data.Feature.values


# In[37]:


X_train,X_test,y_train,y_test = train_test_split(X_data[cols],y,test_size=.3)


# In[38]:


rfr = RandomForestRegressor(n_estimators=25, min_samples_leaf=3, max_features='sqrt',
                          n_jobs=-1)


# In[39]:


rfr.fit(X_train,y_train)


# In[40]:


y_pred = rfr.predict(X_test)


# In[41]:


mean_squared_error(y_pred,y_test)


# In[42]:


mean_absolute_error(y_pred,y_test)


# In[58]:


out = rfr.predict(data_test[cols])


# In[63]:


outdf = pd.DataFrame(data = out,columns=['winPlacePerc'])


# In[64]:


submisson_V2 = pd.concat([test_id,outdf],axis=1)


# In[ ]:





# In[ ]:





# In[ ]:




