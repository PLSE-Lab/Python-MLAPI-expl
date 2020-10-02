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


df = pd.read_csv("../input/insurance.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.isna().sum()


# In[ ]:


num_col = df.select_dtypes(include=np.number).columns
num_col


# In[ ]:


cat_col = df.select_dtypes(exclude=np.number).columns
cat_col


# In[ ]:


encoded_cat_col = pd.get_dummies(df[cat_col])


# In[ ]:


encoded_cat_col


# In[ ]:


df_preprocesssed = pd.concat([df[num_col], encoded_cat_col], axis = 1)


# In[ ]:


df_preprocesssed


# In[ ]:


x = df_preprocesssed.drop(columns=['expenses'])
y = df_preprocesssed[['expenses']]


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state = 1)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model = LinearRegression()


# In[ ]:


model.fit(train_x, train_y)


# In[ ]:


model.intercept_


# In[ ]:


model.coef_


# In[ ]:


predict_train = model.predict(train_x)
predict_test = model.predict(test_x)


# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[ ]:


train_MSE = mean_squared_error(train_y, predict_train)
test_MSE = mean_squared_error(test_y, predict_test)


# In[ ]:


train_MAE = mean_absolute_error(train_y, predict_train)
test_MAE= mean_absolute_error(test_y, predict_test)


# In[ ]:


train_RMSE = np.sqrt(train_MSE)
test_RMSE = np.sqrt(test_MSE)


# In[ ]:


train_MAPE = np.mean(np.abs(train_y, predict_train))
test_MAPE = np.mean(np.abs(test_y, predict_test))


# In[ ]:


print("train_MAE: ",train_MAE)
print("test_MAE: ",test_MAE)


# In[ ]:


print("train_MSE: ",train_MSE)
print("test_MSE: ",test_MSE)


# In[ ]:


print("train_RMSE: ", train_RMSE)
print("test_RMSE: ", test_RMSE)


# In[ ]:


print("train_MAPE: ", train_MAPE)
print("test_MAPE: ", test_MAPE)

