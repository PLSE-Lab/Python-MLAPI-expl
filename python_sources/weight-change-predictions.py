#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns # data visualization

from sklearn.model_selection import train_test_split # train test splitting

from sklearn.preprocessing import StandardScaler # to normalize the values

from sklearn.linear_model import Ridge # Regression machine learning model

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.h


# In[ ]:


data = pd.read_csv('../input/diet_data.csv')
data.head()


# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


print(data.isnull().sum())
sns.heatmap(data.isnull())


# In[ ]:


data = data.drop(['Date'],axis= 1)
data.head()


# In[ ]:


data = data.dropna()
data.head()


# In[ ]:


data.shape


# In[ ]:


data['cals_per_oz'] = data.cals_per_oz.astype('float')
print(data.info())
sns.heatmap(data.isnull())


# In[ ]:


X = data.iloc[:,:12]
Y = data.iloc[:,-1]
X.head()


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 15,test_size = 0.35)


# In[ ]:


print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


# In[ ]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[ ]:


model = Ridge()
model.fit(X_train_scaled,Y_train)


# In[ ]:


model.predict(X_test_scaled)


# In[ ]:


model.score(X_train_scaled,Y_train)


# In[ ]:


model.score(X_test_scaled,Y_test)

