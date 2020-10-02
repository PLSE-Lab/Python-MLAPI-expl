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


# **Importing Libraries**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# **Importing dataset(s)**

# In[ ]:


housing = pd.read_csv('../input/USA_Housing.csv')


# **Looking in the dataset**

# In[ ]:


housing.head()


# In[ ]:


housing.info()


# In[ ]:


housing.describe()


# In[ ]:


housing.rename(columns = {'Avg. Area Income':'income','Avg. Area House Age':'age','Avg. Area Number of Rooms':'room','Avg. Area Number of Bedrooms':'bedroom','Area Population':'population','Price':'price','Address':'address'},inplace=True)


# In[ ]:


housing.columns


# In[ ]:


housing.describe(include = ['O'])


# In[ ]:


x = housing.iloc[:,1:6]
y = housing.iloc[:,0:1]


# **Normalise/Standardise the data**

# In[ ]:


def normalise(x):
    for i in x.columns:
        x[i] = (x[i]-np.min(x[i]))/(np.max(x[i])-np.min(x[i]))
normalise(x)
normalise(y)


# **Creating training and testing data**

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.3,random_state=0)


# Linear Regression

# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)


# In[ ]:


y_pred_lr = lr.predict(x_test)
c = [i for i in range(1500)]
plt.figure(figsize=(16,7))
plt.plot(c,y_test - y_pred_lr,'-')
plt.show()


# In[ ]:


import statsmodels.api as sm
x_train_sm = sm.add_constant(x_train)
x_test_sm = sm.add_constant(x_test)
lr_ols_1 = sm.OLS(y_train,x_train_sm).fit()


# In[ ]:


print(lr_ols_1.summary())


# 84% accuracy

# In[ ]:


x_train_sm.drop('bedroom',inplace=True,axis=1)
import statsmodels.api as sm
lr_ols_1 = sm.OLS(y_train,x_train_sm).fit()
print(lr_ols_1.summary())


# Polynomial Regression

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
lr1 = PolynomialFeatures(degree=3)
x_train_pf = lr1.fit_transform(x_train)
x_test_pf = lr1.fit_transform(x_test)
lr_pf = LinearRegression()
lr_pf.fit(x_train_pf,y_train)


# In[ ]:


y_pred_pf = lr_pf.predict(x_test_pf)


# In[ ]:


plt.figure(figsize=(10,7))
plt.plot(c, y_test - y_pred_pf,'-')


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test,y_pred_pf)
r2 = r2_score(y_test,y_pred_pf)
print(mse)
print(r2)


# Support Vector Regression  

# In[ ]:


from sklearn.svm import SVR
lr_svr = SVR()
lr_svr.fit(x_train,y_train)
y_pred_svr = lr_svr.predict(x_test)


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test,y_pred_svr)
r2 = r2_score(y_test,y_pred_svr)
print(mse)
print(r2)


# Decision Tree Regression

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
lr_dtr = DecisionTreeRegressor()
lr_dtr.fit(x_train,y_train)
y_pred_dtr = lr_dtr.predict(x_test)


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test,y_pred_dtr)
r2 = r2_score(y_test,y_pred_dtr)
print(mse)
print(r2)


# Random Forest Regression

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
lr_rfr = RandomForestRegressor(n_estimators = 100)
lr_rfr.fit(x_train,y_train)
y_pred_rfr = lr_rfr.predict(x_test)


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test,y_pred_rfr)
r2 = r2_score(y_test,y_pred_rfr)
print(mse)
print(r2)

