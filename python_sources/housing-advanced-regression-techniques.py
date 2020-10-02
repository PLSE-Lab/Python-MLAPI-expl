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

df = pd.read_csv('../input/house-prices-advanced-regression-techniques/housetrain.csv')


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


pd.set_option('display.max_rows', 500)
df.dtypes


# In[ ]:


len(df.columns)


# In[ ]:


correlation = df.corr()['SalePrice']
correlation 


# In[ ]:


import matplotlib.pyplot as plt

plt.matshow(df.corr())
plt.show()


# In[ ]:


X = df[['Neighborhood','OverallQual','YearBuilt','ExterCond','TotalBsmtSF','GrLivArea','SalePrice']]


# In[ ]:


X=df[['FullBath','OverallQual','YearBuilt','TotalBsmtSF','GrLivArea','GarageCars']]
y=df['SalePrice']


# In[ ]:


#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#regressor=LinearRegression()
regressor=RandomForestRegressor()
regressor.fit(X,y)


# In[ ]:


y_pred=regressor.predict(X)


# In[ ]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, y_pred) #fix output


# In[ ]:


rmse = np.sqrt(mse)
rmse


# In[ ]:


rmse/y.mean()


# In[ ]:


y_pred = pd.Series(y_pred)


# In[ ]:


y_pred = pd.DataFrame(y_pred)


# In[ ]:


y_pred['id'] = df['Id']
y_pred = y_pred.rename(columns={0: 'SalePrice'})
y_pred = y_pred[['id', 'SalePrice']]


# In[ ]:


y_pred.to_csv('submission.csv', index=False)


# In[ ]:


y_pred.columns

