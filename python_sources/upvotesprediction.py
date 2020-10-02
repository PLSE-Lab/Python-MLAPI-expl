#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import ensemble
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing


# In[2]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[3]:


test = pd.read_csv('../input/test_8i3B3FC.csv')
train = pd.read_csv('../input/train_NIR5Yl1.csv')


# In[4]:


train.head()


# In[5]:


train.isnull().sum()


# In[6]:


test.head()


# In[7]:


test.isnull().sum()


# In[8]:


train.shape


# In[9]:


test.shape


# In[10]:


unique, counts = np.unique(train['Tag'], return_counts=True)
print(unique)
print(counts)


# In[11]:


unique, counts = np.unique(test['Tag'], return_counts=True)
print(unique)
print(counts)


# In[12]:


yTrain=train['Upvotes']
train.drop(labels=['Upvotes','ID','Username'],axis=1,inplace=True)


# In[13]:


ID_test = test['ID']
test.drop(labels=['ID','Username'],axis=1,inplace=True)
train.head()


# In[14]:


test = pd.get_dummies(test)


# In[15]:


train = pd.get_dummies(train)


# In[16]:


train.head()


# In[17]:


test.head()


# In[18]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train[['Reputation','Views','Answers']] = scaler.fit_transform(train[['Reputation','Views','Answers']])
test[['Reputation','Views','Answers']] = scaler.fit_transform(test[['Reputation','Views','Answers']])
train.head(5)


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(train,
yTrain,test_size=0.20, random_state=42)


# In[20]:


from sklearn.linear_model import SGDRegressor

reg = SGDRegressor(loss = 'epsilon_insensitive', verbose=1,eta0=0.1,n_iter=60)

reg.fit(X_train, Y_train)


# In[21]:


model = ensemble.RandomForestRegressor(n_estimators = 100, random_state = 100, verbose=1)
model.fit(X_train, Y_train)


# In[23]:


from sklearn.metrics import mean_squared_error
Y_pred = model.predict(X_valid)
error = mean_squared_error(Y_valid, Y_pred)
print(error)


# In[24]:


model.score(X_train, Y_train)


# In[26]:


Upvotes = pd.Series(np.abs(model.predict(test)), name="Upvotes")

results = pd.concat([ID_test,Upvotes],axis=1)

results.to_csv("submission.csv",index=False)


# In[ ]:




