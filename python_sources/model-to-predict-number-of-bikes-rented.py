#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/bike_share.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.corr()['count']


# There are no null values here. Categorical columns like weather, season, holiday and working day are already encoded.
# 
# Casual and registered user columns are highly correlated with count because count is resultant of sum of those two columns. So, we can drop casual and registered columns since they can overfit the model.
# 
# Also temp and atemp appear to be same just that the normalization factor is different. Let us drop atemp and keep temp column.

# In[ ]:


df.drop(columns=['atemp','casual','registered'],inplace = True)


# In[ ]:


df.head()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.pairplot(df)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


X = df[['season','holiday','workingday','weather','temp','humidity','windspeed']]
y = df['count']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)


# In[ ]:


model = LinearRegression()


# In[ ]:


model.fit(X_train,y_train)


# In[ ]:


model.coef_


# In[ ]:


model.intercept_


# In[ ]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[ ]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[ ]:


train_predict = model.predict(X_train)

mae_train = mean_absolute_error(y_train,train_predict)

mse_train = mean_squared_error(y_train,train_predict)

rmse_train = np.sqrt(mse_train)

r2_train = r2_score(y_train,train_predict)

mape_train = mean_absolute_percentage_error(y_train,train_predict)


# In[ ]:


test_predict = model.predict(X_test)

mae_test = mean_absolute_error(test_predict,y_test)

mse_test = mean_squared_error(test_predict,y_test)

rmse_test = np.sqrt(mean_squared_error(test_predict,y_test))

r2_test = r2_score(y_test,test_predict)

mape_test = mean_absolute_percentage_error(y_test,test_predict)


# In[ ]:


print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('TRAIN: Mean Absolute Error(MAE): ',mae_train)
print('TRAIN: Mean Squared Error(MSE):',mse_train)
print('TRAIN: Root Mean Squared Error(RMSE):',rmse_train)
print('TRAIN: R square value:',r2_train)
print('TRAIN: Mean Absolute Percentage Error: ',mape_train)
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('TEST: Mean Absolute Error(MAE): ',mae_test)
print('TEST: Mean Squared Error(MSE):',mse_test)
print('TEST: Root Mean Squared Error(RMSE):',rmse_test)
print('TEST: R square value:',r2_test)
print('TEST: Mean Absolute Percentage Error: ',mape_test)


# In[ ]:




