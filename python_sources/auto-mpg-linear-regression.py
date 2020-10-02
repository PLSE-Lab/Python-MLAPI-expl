#!/usr/bin/env python
# coding: utf-8

# Linear Regression with auto-mpg.
# 
# Some EDA followed by linear regression.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data=pd.read_csv("../input/autompg-dataset/auto-mpg.csv")
data.head()


# In[ ]:


data.describe()


# In[ ]:


data.shape


# In[ ]:


data.horsepower


# In[ ]:


sum(data.horsepower=='?')


# In[ ]:


data=data[data.horsepower!='?']


# In[ ]:


data.horsepower=data.horsepower.astype('int64')


# In[ ]:


data.describe()


# In[ ]:


data['car name'].value_counts()


# In[ ]:


data['car name'].fillna('dddddd')


# In[ ]:


data['car name']=[i[0] for i in data['car name'].str.split(' ')]


# In[ ]:


data['car name'].unique()


# In[ ]:


data['car name']=data['car name'].replace(['chevrolet','chevy','chevroelt'],'chevrolet')
data['car name']=data['car name'].replace(['volkswagen','vw','vokswagen'],'volkswagen')
data['car name']=data['car name'].replace('maxda','mazda')
data['car name']=data['car name'].replace('toyouta','toyota')
data['car name']=data['car name'].replace('mercedes','mercedes-benz')
data['car name']=data['car name'].replace('nissan','datsun')
data['car name']=data['car name'].replace('capri','ford')


# In[ ]:


len(data['car name'])


# In[ ]:


data.info()


# In[ ]:


plt.hist(data.mpg)


# In[ ]:


sns.pairplot(data)


# In[ ]:


from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


# In[ ]:


data=pd.concat([data,pd.get_dummies(data.origin,prefix='origin')],axis=1)
data.drop('origin',axis=1,inplace=True)


# In[ ]:


data=pd.concat([data,pd.get_dummies(data.cylinders,prefix='cylinders')],axis=1)
data.drop('cylinders',axis=1,inplace=True)


# In[ ]:


data=pd.concat([data,pd.get_dummies(data['model year'],prefix='year')],axis=1)
data.drop('model year',axis=1,inplace=True)


# In[ ]:


data.head(7)


# In[ ]:


data[['displacement','horsepower','weight','acceleration']]=StandardScaler().fit_transform(data[['displacement','horsepower','weight','acceleration']])


# In[ ]:


data=pd.concat([data,pd.get_dummies(data['car name'],prefix='car')],axis=1)
data.drop('car name',axis=1,inplace=True)


# In[ ]:


data.shape


# In[ ]:


y = data.pop('mpg')
X = data


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# In[ ]:


lr=LinearRegression()
lr.fit(X_train,y_train)


# In[ ]:


y_pred=lr.predict(X_test)
mean_squared_error(y_pred,y_test)


# In[ ]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

