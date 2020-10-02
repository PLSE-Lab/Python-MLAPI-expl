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


dataset = pd.read_csv('../input/startup-logistic-regression/50_Startups.csv')


# In[ ]:


dataset.columns


# In this dataset we are able to see there are five columns,
# 1. R&D Spend
# 1. Administration
# 1. Marketing spend
# 1. State
# 1. Profit.
# 
# From this dataset we are going to analysis how the these columns (R&D spend,Administration,Marketing spend,State) will impact in determining the Profit.
# 
# Here Profit is the dependent varialble.

# In[ ]:


dataset.isnull().sum().sort_values(ascending=False)


# So there is no missing values in the dataset then there is no need for data imputation.
# 
# We will assign the **dependent and independent variables**.

# In[ ]:


X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4:5].values


# In[ ]:


dataset.head()


# We are going to check the linear relationship in the dataset. In our independent varialbe there is categorical varible so we will use the encoding technique.

# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
#onehotencoder = OneHotEncoder(categories='auto')
X = onehotencoder.fit_transform(X).toarray()


# In[ ]:


X = X[:,1:]


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)  


# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[ ]:


y_pred = regressor.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


y_test

