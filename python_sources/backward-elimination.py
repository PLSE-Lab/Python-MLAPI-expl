#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


data=pd.read_csv('../input/50_Startups.csv')
data.head(4)


# In[ ]:


data.State.value_counts()


# State is only categorical variable with 3 labels. Label encoder is used only when we have <=2 labels.Because of 3 labels I am using on hot encoder. Before going that process I am checking the null values. 

# In[ ]:


data.isnull().sum()


# Its nice we don't have any null values. ;)

# In[ ]:


data=pd.get_dummies(data)


# In[ ]:


data.head(4)


# In[ ]:


X=data.drop(['Profit'],axis=1)
X.head(4)


# In[ ]:


Y=data.Profit
Y.head(4)


# In[ ]:


from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[ ]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)


# In[ ]:


Y_pred=regressor.predict(X_test)


# In[ ]:


from sklearn.metrics import mean_squared_error
from math import sqrt

mse=round((mean_squared_error(Y_test,Y_pred))/100, 2)
rmse = round((sqrt(mse))/100 ,2)
mse ,rmse


# In order to reduce my MSE value I want to go for feature selection by Backward Elimination.

# In[ ]:


import statsmodels.api as sm
X=sm.add_constant(X)
model=sm.OLS(Y,X).fit()
model.summary()


# Backward Elimination: Statsmodels.formula library is required. Selecting P value as < 0.05
# ![1](http://3.bp.blogspot.com/-Q5NbRCOwVQc/WVCRFGON1QI/AAAAAAAABsc/Nzu8UtbejwoqFf6FYWhUGuWpMd-jDfAtwCLcBGAs/s1600/Capture5.PNG)

# In[ ]:


X=X.drop(['Administration'],axis=1)
model=sm.OLS(Y,X).fit()
model.summary()


# In[ ]:


X=X.drop(['Marketing Spend'],axis=1)
model=sm.OLS(Y,X).fit()
model.summary()


# In[ ]:


from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[ ]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)


# In[ ]:


Y_pred=regressor.predict(X_test)


# In[ ]:


mse=round((mean_squared_error(Y_test,Y_pred))/100, 2)
rmse = round((sqrt(mse))/100 ,2)
mse ,rmse


# **Please UPVOTE for Encouragement**
