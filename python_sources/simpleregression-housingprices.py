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


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# In[ ]:


dataset = pd.read_csv('../input/kc_house_data.csv')
dataset.columns


# In[ ]:


dataset['lat'].unique()
dataset['zipcode'].unique()


#zipcode, lat, long all indicate the location of house so i am going to drop lat, long 


# In[ ]:


#we donot need id. so lets drop it
dataset = dataset.drop(['id','lat','long'],axis = 1)

#missing data
# the count is same for all columns which means there is no missing data.
[dataset.iloc[i,j] for i,j in zip(*np.where(pd.isnull(dataset)))]


# In[ ]:


#categorical data
#lets find out the categorical data
for col in dataset.columns:
    print(col,dataset[col].unique(), sep = "-")


# In[ ]:





# In[ ]:


dataset.corr()


# In[ ]:


#sqft_living has high correlation with sqft_living15
#sqft_lot has high correlation with sqft_lot15
#sqft_above has high correlation with sqft_living
#lets drop column sqft_living15,sqft_lot15,sqft_above
dataset = dataset.drop(['sqft_living15','sqft_lot15','sqft_above'],axis = 1)


# In[ ]:


dataset.columns
dataset.head(10)


# In[ ]:


dataset.hist()
plt.rcParams['figure.figsize'] = [60, 20]
plt.show()


# In[ ]:


dataset.dtypes


# In[ ]:


dataset.date = dataset.date.str[:8]


# In[ ]:


dataset.head()


# In[ ]:


dataset['date'] = pd.to_numeric(dataset['date'])


# In[ ]:


dataset.dtypes


# In[ ]:


X = pd.concat([dataset.iloc[:,0],dataset.iloc[:,2:]], axis=1)


# In[ ]:


X


# In[ ]:


y = dataset.iloc[:,1]


# In[ ]:


y


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[ ]:


X_train


# In[ ]:


#stat model

import statsmodels.formula.api as sm
lm1 = sm.OLS(endog = y_train, exog = X_train).fit()

# print the coefficients
lm1.summary()


# In[ ]:


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
regressor.score(X_test,y_test)


# In[ ]:




