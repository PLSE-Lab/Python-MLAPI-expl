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


import pandas as pd
import numpy as nm
from matplotlib import pyplot as plt


# In[ ]:


data=pd.read_csv('../input/house.csv')


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.rename(columns ={'price': 'SalePrice'}, inplace =True)
data.head()


# In[ ]:


Y = data.SalePrice.values
feature_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
       'sqft_living15', 'sqft_lot15']
X=data[feature_cols]


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(X, Y, random_state=3)


# In[ ]:


reg = LinearRegression()


# In[ ]:


reg.fit(x_train, y_train)


# In[ ]:


accuracy = reg.score(x_test, y_test)


# In[ ]:


"Accuracy: {}%".format(int(round(accuracy * 100)))


# In[ ]:


data.shape


# In[ ]:


plt.scatter(data.sqft_living,data.SalePrice)
plt.title("price vs square feet")
plt.xlabel("sqare feet")
plt.ylabel("price")


# In[ ]:




