#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')
data.head()


# Splitting the variables (dependent and independent)
# 

# In[ ]:


x = data.iloc[:,:-1].values
y = data.iloc[:,1:2].values


# In[ ]:


# checking null values in dataset
# we don't have miss data
data.isna().sum()


# In[ ]:


# checking the variables
plt.figure(figsize=(10,8))
sns.regplot( x = 'YearsExperience', y = 'Salary', data = data)
plt.title('Scatter with linear line')


# now we have to split the data into train and test
# 

# In[ ]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y, random_state = 0)


# In[ ]:


train_x.shape, train_y.shape, test_x.shape, test_y.shape 


# Implementing linear regression
# 

# In[ ]:


# importing the data
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score


# In[ ]:


# creating instance of linear regression
lr = LR()


# In[ ]:


# fitting the model
lr.fit(train_x, train_y)


# In[ ]:


# Predicting over the Train Set and calculating error
train_predict = lr.predict(train_x)
k = mae(train_predict, train_y)
print('Test Mean absolut error is ', k)


# In[ ]:


# Predicting over the test set and calculating error
test_predict = lr.predict(test_x)
k = mae(test_predict, test_y)
print('Test Mean Absolut error is ', k)


# In[ ]:


r2_square = r2_score(test_y, test_predict)
print('R2_Square : ',r2_square)
print('Accurarcy of model : ',r2_square*100)


# In[ ]:


print(lr.intercept_)
print(lr.coef_)


# **the equation for salary is : SALARY = 9379.71049195*YearsExperience + 26986.69131674**
# 
#                             

# In[ ]:




