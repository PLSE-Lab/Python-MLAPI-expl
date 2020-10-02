#!/usr/bin/env python
# coding: utf-8

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


df=pd.read_csv("/kaggle/input/heights-and-weights/data.csv")


# In[ ]:


df.head(5)


# In[ ]:


#idnetify if any null values indataset
df.isnull().any()


# In[ ]:


#to understand how data are corelated
import seaborn as sb
sb.scatterplot(x=df['Weight'],y=df['Height'])


# from scatterplot we can see data are so perfectly linear and positive co-related

# In[ ]:


#to fins size of the dataset
df.shape


# In[ ]:


#we will split our x and y variables
x=df.iloc[:,:1]
y=df.iloc[:,-1]


# In[ ]:


print(x.shape)
print(y.shape)


# In[ ]:


#data set is too small but we will try to divide them and use 30% of data for testing purpose
from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[ ]:


#printing sizes of data set
print("Size of x_train is {}".format(x_train.shape))
print("Sie of x_test is {}".format(x_test.shape))
print("Sie of y_train is {}".format(y_train.shape))
print("Sie of y_test is {}".format(y_test.shape))


# In[ ]:


# importing Simple linear regression algo
from sklearn.linear_model import LinearRegression


# In[ ]:


le=LinearRegression()


# In[ ]:


le.fit(x_train,y_train)


# In[ ]:


y_pred=le.predict(x_test)


# In[ ]:


print(y_pred)


# In[ ]:


print(y_test)


# In[ ]:


#plotting a grap b/w observed value and predicted
import matplotlib.pyplot as plt
sb.scatterplot(x=y_test,y=y_pred)
plt.title("Observed Vs Predited")
plt.xlabel("y_test")
plt.ylabel("y_pred")


# In[ ]:


#to check accuracy we will use r2 value
from sklearn.metrics import r2_score


# In[ ]:


print(r2_score(y_test,y_pred))


# In[ ]:


# or we can also use score in LinearRegression pkg
le.score(x_test,y_test)

