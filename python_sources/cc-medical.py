#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('/kaggle/input/insurance/insurance.csv')


# In[ ]:


df.head(2)


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.distplot(df.age, kde = False, bins = 40);


# In[ ]:


df.age.value_counts().head()


# In[ ]:


df.columns


# In[ ]:


sns.countplot('sex', data = df)


# In[ ]:


df.columns


# In[ ]:



sns.distplot(df.bmi, kde = False, bins = 40);


# In[ ]:


sns.countplot('children', data = df);


# In[ ]:


sns.countplot('smoker', data = df);


# In[ ]:


df.columns


# In[ ]:


sns.countplot('region', data = df);


# In[ ]:


sns.distplot(df.charges, kde = False, bins = 40);


# In[ ]:


# Create Dummy Variables


#  # **'sex', 'children', 'smoker', 'region' can be converted to dummies**

# In[ ]:


sex = pd.get_dummies(df.sex, drop_first= True)


# In[ ]:


children = pd.get_dummies(df.children, drop_first= True)


# In[ ]:


smoker = pd.get_dummies(df.smoker, drop_first= True)


# In[ ]:


region = pd.get_dummies(df.region, drop_first= True)


# In[ ]:


train = pd.concat([df,sex, children, smoker, region], axis = 1)


# In[ ]:


train.head(2)


# In[ ]:


train.drop(['sex', 'children', 'smoker', 'region'], axis = 1, inplace = True)


# In[ ]:


train


# In[ ]:





# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = train.drop('charges', axis = 1)


# In[ ]:


y = train.charges


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


X_train


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train, y_train)


# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


predictions


# In[ ]:


from sklearn import metrics 
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:


print(lm.score(X_test,y_test))


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


RFR = RandomForestRegressor()


# In[ ]:


RFR.fit(X_train, y_train)


# In[ ]:


print(RFR.score(X_test,y_test))


# In[ ]:


Rpredictions = RFR.predict(X_test)


# In[ ]:


from sklearn import metrics 
print('MAE:', metrics.mean_absolute_error(y_test, Rpredictions))
print('MSE:', metrics.mean_squared_error(y_test, Rpredictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, Rpredictions)))


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor


# In[ ]:


KNN = KNeighborsRegressor()


# In[ ]:


KNN.fit(X_train, y_train)


# In[ ]:


print(KNN.score(X_test,y_test))


# In[ ]:


Kpredictions = KNN.predict(X_test)


# In[ ]:


Kpredictions


# In[ ]:


print('MAE:', metrics.mean_absolute_error(y_test, Kpredictions))
print('MSE:', metrics.mean_squared_error(y_test, Kpredictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, Kpredictions)))


# In[ ]:





# # Random Forrest is the best model

# In[ ]:




