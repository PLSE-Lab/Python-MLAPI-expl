#!/usr/bin/env python
# coding: utf-8

# # Sucide rate analysis using Decesion tree 

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree
import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.


# In[3]:


file_path = '../input/master.csv'
df = pd.read_csv(file_path)
df.head() # print the first few lines 


# In[4]:


df.describe() # print the statistics of the datasets 


# In[5]:


df.columns # print columns of the datasets


# In[6]:


# select columns from the dataset and prepare a new datasets 
col = ['year', 'population',
       'suicides/100k pop',
        'gdp_per_capita ($)']
df1 = df[col]
print(df1.head())
y = df['suicides_no']
print(y.head())


# In[ ]:





# In[7]:


from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split


# In[8]:


# devide the dataset into the 
train_x, test_x, train_y, test_y = train_test_split(df1,y, test_size = 0.2)
print(train_x.shape, train_y.shape)
print(test_y.shape, test_x.shape)
print(train_x.head())


# In[9]:


train_x.head()


# In[10]:


model = DecisionTreeRegressor(random_state =0)
model.fit(train_x,train_y)


# In[11]:


predict = model.predict(test_x)


# In[25]:


import matplotlib.pyplot as plt
plt.figure(1)
plt.figure(figsize = (20,10))
plt.plot(np.arange(len(predict)),predict, 'o', label = "predicted values")
plt.title('predicted and actual suicide rate')
plt.xlabel("number of samples ")
#plt.figure(2)
plt.plot(np.arange(len(predict)),test_y.values, label = "actual values")
plt.legend(loc = 'best')
#plt.title('actual suicide rate')


# In[16]:


# accuracy score 
from sklearn.metrics import accuracy_score
accuracy_score(test_y, predict)


# In[17]:


# mean absolute error 
from sklearn.metrics import mean_absolute_error 
mean_absolute_error(test_y, predict)


# In[18]:


# mean absolute percentage error 
def mape(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[19]:


mape(test_y, predict)


# In[ ]:





# In[ ]:




