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


import matplotlib.pyplot as plt 
import seaborn as sns


# In[ ]:


df =pd.read_csv('/kaggle/input/usa-housing/USA_Housing.csv')


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# # EDA

# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


sns.pairplot(df)


# In[ ]:


sns.heatmap(df.corr(),annot = True)


# ** we like to get rid of the address column since it does not have anything that can be used in linear regression model  **

# In[ ]:


df.columns


# In[ ]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']


# **now that we have selected the feature and target variables we would like to split the data into training set for building the model and testing set for testing the model**

# # linear regression

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train , X_test, y_train ,y_test = train_test_split(
    X ,y ,test_size = 0.4, random_state = 101
)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train , y_train )


# **next step is to evaluate our model **

# In[ ]:


print("intercept")
print(lm.intercept_)


# In[ ]:


print("coefficeints")
print(lm.coef_)


# # **predictions **

# In[ ]:


predictions = lm.predict(X_test)
predictions


# **since we have the results of actual values we would like to know how accurate is our prediction**

# In[ ]:


plt.scatter(y_test,predictions)


# In[ ]:


# we would like to visualize the preditions
sns.distplot(y_test - predictions)


# **our histogram of residual is normally distributed which implies that our model was the correct choice **

# #### evaluating the model

# In[ ]:


from sklearn.metrics import mean_absolute_error , mean_squared_error


# In[ ]:


mean_absolute_error(y_test , predictions)


# In[ ]:


mean_squared_error(y_test , predictions)


# In[ ]:




