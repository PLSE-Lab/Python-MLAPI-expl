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


data=pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')


# In[ ]:


data


# In[ ]:


data.isnull().sum()


# In[ ]:


data.dropna(inplace=True,axis=0)


# In[ ]:


data.shape


# In[ ]:


from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
data.ocean_proximity=enc.fit_transform(data.ocean_proximity)


# In[ ]:


data


# In[ ]:


x=data.iloc[:,[0,1,2,3,4,5,6,7,9]]


# In[ ]:


y=data.median_house_value


# In[ ]:


x


# In[ ]:


y


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.05,random_state=10)


# **Modelling**

# **Linear Regression**

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


model=LinearRegression()


# In[ ]:


model.fit(x_train,y_train)


# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


r2_score(y_test,model.predict(x_test))


# **RANDOM FOREST CLASSIFIER**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


model2=RandomForestRegressor()


# In[ ]:


model2.fit(x_train,y_train)


# In[ ]:


r2_score(y_test,model2.predict(x_test))


# In[ ]:




