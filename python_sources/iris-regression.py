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

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/Iris.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.keys()


# In[ ]:


features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = X[features]


# In[ ]:


a = df['Species'].values
print(a)


# In[ ]:


X = df.iloc[0:100,[1,2,3,4]]
print(X.head())


# In[ ]:


y = df.iloc[0:100,5]
print(y)


# In[ ]:


y=np.where(y=='Iris-setosa',-1,1)


# In[ ]:


print(y.shape)


# In[ ]:


for x in range y:
    if y[x] == 'Iris-setosa':
        y[x]=1
    else:
        y[x]=-1


# In[ ]:


y.shape()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42)


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


y = y.iloc[0:100,1]


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


print(X_train.shape)
print(y_train.shape)


# In[ ]:


print(X_train.head())


# In[ ]:


reg= LinearRegression()


# In[ ]:


reg.fit(X_train,y_train)


# In[ ]:


a=reg.score(X_train,y_train)
print(a)
print(reg.score(X_test,y_test))


# In[ ]:




