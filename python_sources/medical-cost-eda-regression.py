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


# **Importing dataset**

# In[ ]:


dataset = pd.read_csv('../input/insurance.csv')


# **Getting some information from the dataset : **

# In[ ]:


dataset.shape


# In[ ]:


dataset.columns


# In[ ]:


dataset.info()


# All values are available. No NaN values.

# In[ ]:


dataset.describe()


# In[ ]:


dataset.describe(include = ['O'])


# In[ ]:


dataset['region'].unique()


# In[ ]:


dataset['sex'].unique()


# In[ ]:


dataset['smoker'].unique()


# We need to encode these categorize variables...

# In[ ]:


#you can do it by mapping
dataset['sex'] = dataset['sex'].map({'male':1,'female':0})

#you can do by looping
dataset.smoker = [1 if each=='yes' else 0 for each in dataset.smoker]


# Lets check our data first, so to see what we can do further. Keep in mind that region column is still categorical...

# In[ ]:


dataset.head(10)


# In[ ]:


dataset.describe()


# Things to do :
#     1. normalise charges columns
#     2. normalise bmi column
#     3. Label encode the region column
#     4. Visualise children column

# In[ ]:


dataset_normal = dataset.copy()
dataset_normal['bmi'] = (dataset_normal['bmi'] - np.min(dataset_normal['bmi']))/(np.max(dataset_normal['bmi']) - np.min(dataset_normal['bmi']))
dataset_normal['charges'] = (dataset_normal['charges'] - np.min(dataset_normal['charges']))/(np.max(dataset_normal['charges']) - np.min(dataset_normal['charges']))


# In[ ]:


dataset_normal.head(10)


# In[ ]:


dummy = pd.get_dummies(dataset_normal['region'],drop_first = True)


# In[ ]:


dataset_normal = pd.concat([dataset_normal,dummy],axis=1)


# In[ ]:


dataset_normal.drop('region',axis=1,inplace = True)


# In[ ]:


dataset_normal.head(15)


# Getting ready the test and train data:

# In[ ]:


x = dataset_normal.copy()
x.drop('charges',axis=1,inplace=True)


# In[ ]:


y = dataset_normal.iloc[:,5:6]


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)


# **multiple linear regression**

# In[ ]:


from sklearn.linear_model import LinearRegression
lr_mlr = LinearRegression()
lr_mlr.fit(x_train,y_train)


# In[ ]:


y_pred_lr = lr_mlr.predict(x_test)
plt.figure(figsize=(10,7))
c = [i for i in range(402)]
plt.plot(c,y_test-y_pred_lr,'-')
plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_test,y_pred_lr)
r2 = r2_score(y_test,y_pred_lr)
print(mse)
print(r2)


# 79% accurate model. Now,lets check with OLS mothod

# In[ ]:


import statsmodels.api as sm
x_train_sm = sm.add_constant(x_train)
lr_sm_1 = sm.OLS(y_train,x_train_sm).fit()
print(lr_sm_1.summary())


# Removing sex column

# In[ ]:


x_train_sm.drop('sex',axis=1,inplace=True)
import statsmodels.api as sm
lr_sm_2 = sm.OLS(y_train,x_train_sm).fit()
print(lr_sm_2.summary())


# Removing northwest column

# In[ ]:


x_train_sm.drop('northwest',axis=1,inplace=True)
import statsmodels.api as sm
lr_sm_3 = sm.OLS(y_train,x_train_sm).fit()
print(lr_sm_3.summary())


# removing southwest and southeast column

# In[ ]:


x_train_sm.drop(['southeast','southwest'],axis=1,inplace=True)
import statsmodels.api as sm
lr_sm_4 = sm.OLS(y_train,x_train_sm).fit()
print(lr_sm_4.summary())


# **Heading further to plynomial Regression-->**

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
lr_pr = PolynomialFeatures(degree=3)
x_train_poly = lr_pr.fit_transform(x_train)
x_test_poly = lr_pr.fit_transform(x_test)
lr_pr1 = LinearRegression()
lr_pr1.fit(x_train_poly,y_train)


# In[ ]:


y_pred_poly = lr_pr1.predict(x_test_poly)


# In[ ]:


c = [i for i in range(402)]
plt.figure(figsize=(10,7))
plt.plot(c,y_test-y_pred_poly,'-')
plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test,y_pred_poly)
r2 = r2_score(y_test,y_pred_poly)
print(mse)
print(r2)


# 85% accuracy . Gorgeous...

# Support Vector Regression

# In[ ]:


from sklearn.svm import SVR
lr_svr = SVR(kernel='linear')
lr_svr.fit(x_train,y_train)


# In[ ]:


y_pred_svr = lr_svr.predict(x_test)


# In[ ]:





# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test,y_pred_svr)
r2 = r2_score(y_test,y_pred_svr)
print(mse)
print(r2)


# 77% accuracy...

# Decision Tree Regression

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
lr_dtr = DecisionTreeRegressor(random_state=0)
lr_dtr.fit(x_train,y_train)


# In[ ]:


y_pred_dtr = lr_dtr.predict(x_test)
plt.figure(figsize=(10,7))
plt.plot(c,y_test,'-')
plt.plot(c,y_pred_dtr,'-o')
plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test,y_pred_dtr)
r2 = r2_score(y_test,y_pred_dtr)
print(mse)
print(r2)


# 71% accurate...

# Random Forest Regression

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
lr_rfr = RandomForestRegressor(n_estimators=100,max_depth=7,random_state=100)
lr_rfr.fit(x_train,y_train)


# In[ ]:


y_pred_rfr = lr_rfr.predict(x_test)
plt.figure(figsize=(10,7))
plt.plot(c,y_test,'-')
plt.plot(c,y_pred_rfr,'-o')
plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test,y_pred_rfr)
r2 = r2_score(y_test,y_pred_rfr)
print(mse)
print(r2)


# 87% accurate.

# In[ ]:





# > **Upvote if you like this regression**
