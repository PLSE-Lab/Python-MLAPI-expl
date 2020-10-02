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


ls


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sb
import sklearn 
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


df_train = pd.read_csv('/kaggle/input/dasprodatathon/train.csv')
df_test = pd.read_csv('/kaggle/input/dasprodatathon/test.csv')


# In[ ]:


print('Dtrain')
display(df_train)

print('Dtest')
display(df_test)


# In[ ]:


display(df_train.describe())


# In[ ]:



df_train.head()


# In[ ]:


sb.pairplot(df_train)


# In[ ]:


len(df_train)


# In[ ]:


df_train.corr()


# In[ ]:


df_train.corr().style.background_gradient().set_precision(2)


# In[ ]:


clf = LinearRegression()


# In[ ]:


x = df_train[['Living Area', 'Above the Ground Area','Neighbors Living Area','Bathrooms', 'Grade', 'Basement Area','Latitude','Longitude','Year Built','Condition','Bedrooms','Floors','Waterfront','View','Year Renovated','Neighbors Total Area','Total Area','Zipcode']]
y = df_train['Price']


# In[ ]:


x


# In[ ]:


y


# In[ ]:


len(df_test)


# In[ ]:


x, x_t, y, y_t = train_test_split(x,y, test_size=0.1, random_state=4)


# In[ ]:


clf.fit(x,y)


# In[ ]:


print(clf.coef_)


# In[ ]:


print(clf.intercept_)


# In[ ]:


clf.score(x,y)


# In[ ]:


clf.score(x_t,y_t)


# In[ ]:



id_test = df_test['ID']
x_test = df_test[['Living Area', 'Above the Ground Area','Neighbors Living Area','Bathrooms', 'Grade', 'Basement Area','Latitude','Longitude','Year Built','Condition','Bedrooms','Floors','Waterfront','View','Year Renovated','Neighbors Total Area','Total Area','Zipcode']]


# In[ ]:


y_pdc = clf.predict(x_test)
display(y_pdc)


# In[ ]:


est = ensemble.GradientBoostingRegressor(n_estimators=500,max_depth=4, random_state=8, min_samples_split = 10, learning_rate = 0.1,validation_fraction=0.1, loss = 'huber').fit(x,y)
est.score(x,y)


# In[ ]:


aw = est.score(x_t, y_t)
aw = aw*100
print(aw)


# In[ ]:


y_pred = est.predict(x_test)
display(y_pred)


# In[ ]:


df_sbmt2 = pd.DataFrame()
df_sbmt2['ID'] = id_test
df_sbmt2['Price'] = y_pred


# In[ ]:


df_sbmt2.head()


# In[ ]:


df_sbmt2.to_csv('AKB_w_linearreg.csv', index=False)


# In[ ]:


df_sbmt = pd.DataFrame()
df_sbmt['ID'] = id_test
df_sbmt['Price'] = y_pdc


# In[ ]:


df_sbmt.head()


# In[ ]:


len(df_test)


# In[ ]:


df_sbmt.to_csv('AKB_cobalagi_linearreg.csv', index=False)


# In[ ]:




