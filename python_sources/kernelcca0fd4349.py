#!/usr/bin/env python
# coding: utf-8

# In[4]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# Any results you write to the current directory are saved as output.


# In[5]:


#test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
traincorr = train.corr()['SalePrice']
traincorr


# In[6]:


used_columns = ["OverallQual", "GrLivArea", "GarageCars", "SalePrice"]
base = pd.read_csv("../input/train.csv", usecols=used_columns )
base.head()


# In[7]:


from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
base[[
    "OverallQual", 
    "GrLivArea", 
    "GarageCars"
 ]] = scaler_x.fit_transform(base[[
    "OverallQual", 
    "GrLivArea", 
    "GarageCars"
 ]])
base.head()


# In[8]:


scaler_y = StandardScaler()
base[['SalePrice']] = scaler_y.fit_transform(base[['SalePrice']])
base.head()


# In[9]:


x = base.drop('SalePrice', axis=1)
y = base.SalePrice


# In[10]:


x.head()


# In[11]:


y.head()


# In[12]:


prev_columns = used_columns[0:3]
prev_columns


# In[13]:


import tensorflow as tf
columns = [tf.feature_column.numeric_column(key=c) for c in prev_columns]
columns


# In[14]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[15]:


x_train.shape


# In[16]:


x_test.shape


# In[17]:


function_train = tf.estimator.inputs.pandas_input_fn(x = x_train, y = y_train,
                                                        batch_size=32,
                                                        num_epochs=None,
                                                        shuffle=True)


# In[18]:


function_test = tf.estimator.inputs.pandas_input_fn(x = x_test, y = y_test,
                                                        batch_size=32,
                                                        num_epochs=100,
                                                        shuffle=False)


# In[19]:


regressor = tf.estimator.LinearRegressor(feature_columns=columns, model_dir='novo')


# In[20]:


regressor.train(input_fn=function_train, steps=100)


# In[21]:


metrics_train = regressor.evaluate(input_fn=function_train, steps=100)


# In[22]:


metrics_test = regressor.evaluate(input_fn=function_test, steps=100)


# In[23]:


metrics_train


# In[24]:


metrics_test


# In[25]:


function_predict = tf.estimator.inputs.pandas_input_fn(x = x_test, shuffle=False)


# In[26]:


predict = regressor.predict(input_fn=function_predict)
list(predict)


# In[27]:


predict_values = [] 
for p in regressor.predict(input_fn=function_predict):
    predict_values.append(p['predictions'])


# In[28]:


predict_values = np.asarray(predict_values).reshape(-1,1)
predict_values


# In[29]:


predict_values = scaler_y.inverse_transform(predict_values)
predict_values


# In[30]:


y_test2 = scaler_y.inverse_transform(y_test.values.reshape(-1,1))
y_test2


# In[31]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(train['GrLivArea'], train['SalePrice'])


# In[32]:


plt.scatter(train['OverallQual'], train['SalePrice'])


# In[33]:


plt.scatter(train['GarageCars'], train['SalePrice'])


# In[34]:





# In[ ]:




