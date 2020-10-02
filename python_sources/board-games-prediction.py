#!/usr/bin/env python
# coding: utf-8

# ****Board Games Prediction Data ****
# Predict avg ratings using the Board Games data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[ ]:


data=pd.read_csv('../input/board-games-prediction-data/games.csv')
data.head()


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


plt.hist(data['average_rating'])
plt.show


# Lest's see why there are Zero Rating 

# In[ ]:


print(data[data['average_rating']==0].iloc[0])


# In[ ]:


data=data[data['users_rated']>0]
data=data.dropna(axis=0)
data.head()


# In[ ]:


data.shape


# In[ ]:


data.corr()


# In[ ]:


plt.hist(data['average_rating'])
plt.show


# In[ ]:


data.head()


# In[ ]:


X=data.drop(['id','name','type','average_rating'],axis=1)
y=data['average_rating']


# In[ ]:


X.head()


# In[ ]:


standardscaler=StandardScaler()
X=standardscaler.fit_transform(X)


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[ ]:


predict=lr.predict(x_test)  
from sklearn.metrics import mean_squared_error
mean_squared_error(predict,y_test)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(random_state=7)
dt.fit(x_train,y_train)


# In[ ]:


dt_predict=dt.predict(x_test)
mean_squared_error(y_test,dt_predict)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=10)
rf.fit(x_train,y_train)


# In[ ]:


rf_predict=rf.predict(x_test)
mean_squared_error(y_test,rf_predict)


# In[ ]:




