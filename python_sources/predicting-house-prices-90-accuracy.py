#!/usr/bin/env python
# coding: utf-8

# **
# 
# Predicting House Prices using Linear Regression and XGBoost
# -----------
# 
# **

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score


# In[ ]:


df =pd.read_csv('../input/kc_house_data.csv')


# In[ ]:


df.head(5)


# In[ ]:


data = pd.concat([df['sqft_living'],df['price']],axis = 1)
plot = data.plot.scatter(x = 'sqft_living',y = 'price')
plot.set_xlabel("Sqft Living")
plot.set_ylabel("House Price")
plot.axes.set_title("Sq ft Living area and House Prices")


# In[ ]:


feature_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
       'sqft_living15', 'sqft_lot15']


# In[ ]:


X = df[feature_cols]


# In[ ]:


y = df['price']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=13)


# In[ ]:


linearModel = LinearRegression()


# In[ ]:


linearModel.fit(X_train,y_train)


# In[ ]:


linearModel.coef_


# In[ ]:


accuracy = linearModel.score(X_train,y_train)
"Accuracy on Train Data: {}%".format(int(round(accuracy * 100)))


# In[ ]:


accuracy = linearModel.score(X_test,y_test)
"Accuracy on Test Data: {}%".format(int(round(accuracy * 100)))


# As we can see our accuracy using Simple Linear Regression is 70% which is pretty low.

# In[ ]:


#Setting up XGBoost Parameters
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.25, gamma=0, subsample=0.75,colsample_bytree=1, max_depth=5)


# In[ ]:


xgb.fit(X_train,y_train)


# In[ ]:


predictions = xgb.predict(X_test)
print(explained_variance_score(predictions,y_test))


# Achieving an accuracy of 90% using XGBoost.
