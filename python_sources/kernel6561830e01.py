#!/usr/bin/env python
# coding: utf-8

# In[59]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[60]:


houses = pd.read_csv('../input/kc_house_data.csv')


# In[72]:


houses.date = houses.date.astype('datetime64')
houses['year'] = houses.date.map(lambda x: x.year)
houses['month'] = houses.date.map(lambda x: x.month)


# In[ ]:





# In[79]:


corr = houses.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr, annot=True)


# In[80]:


X = houses.drop(columns=['id', 'price', 'date', 'condition', 'long', 'sqft_lot15', 'zipcode', 'yr_built', 'sqft_lot'])
y = houses['price']


# In[81]:


from sklearn.model_selection import train_test_split


# In[82]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[83]:


X_train.head()


# ### LinearRegression

# In[84]:


from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
model.fit(X_train, y_train)
pred = model.predict(X_test)
plt.scatter(y_test, pred)

from sklearn import metrics

print('MAE: ', metrics.mean_absolute_error(y_test, pred))
print('MSE: ', metrics.mean_squared_error(y_test, pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# ### RandomForestRegressor

# In[85]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
pred =model.predict(X_test)
plt.scatter(y_test, pred)

from sklearn import metrics

print('MAE: ', metrics.mean_absolute_error(y_test, pred))
print('MSE: ', metrics.mean_squared_error(y_test, pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# ### BagginRegressor

# In[86]:


from sklearn.ensemble import BaggingRegressor
model = BaggingRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)
plt.scatter(y_test, pred)

from sklearn import metrics

print('MAE: ', metrics.mean_absolute_error(y_test, pred))
print('MSE: ', metrics.mean_squared_error(y_test, pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# In[87]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)
plt.scatter(y_test, pred)

from sklearn import metrics

print('MAE: ', metrics.mean_absolute_error(y_test, pred))
print('MSE: ', metrics.mean_squared_error(y_test, pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# In[88]:


from sklearn import tree
model = tree.DecisionTreeRegressor()
model.fit(X_train, y_train)
pred = model.predict(X_test)
plt.scatter(y_test, pred)

from sklearn import metrics

print('MAE: ', metrics.mean_absolute_error(y_test, pred))
print('MSE: ', metrics.mean_squared_error(y_test, pred))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, pred)))


# In[ ]:




