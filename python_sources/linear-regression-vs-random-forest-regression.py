#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# In[3]:


data = pd.read_csv('../input/boston_housing.csv')


# In[4]:


data.head()


# In[5]:


data.describe()


# In[6]:


data['medv'].describe()


# In[7]:


data.info()


# In[8]:


df_scaled = pd.DataFrame(preprocessing.scale(data), columns=data.columns)
df_scaled.head()


# In[9]:


df_scaled.describe()


# In[10]:


plt.figure(figsize=(50,50))
sns.pairplot(df_scaled,y_vars='medv',x_vars=df_scaled.columns[:-1])


# In[11]:


plt.figure(figsize=(50,50))
sns.pairplot(data,y_vars='medv',x_vars=data.columns[:-1])


# In[12]:


plt.figure(figsize=(20,20))
sns.heatmap(df_scaled.corr(),annot=True,fmt='.1f',linewidths=2)


# In[13]:


X = df_scaled.drop('medv',axis=1)
y = data['medv']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[14]:


linmodel = LinearRegression()
linmodel.fit(X_train,y_train)
linpred = linmodel.predict(X_test)


# In[15]:


plt.scatter(y_test,linpred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[16]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, linpred))
print('MSE:', metrics.mean_squared_error(y_test, linpred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, linpred)))


# In[17]:


from sklearn.linear_model import Lasso
alpha_ridge = [-3,-2,-1,1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1,1.5, 2,3,4, 5, 10, 20, 30, 40]
temp_mae = {}
temp_mse = {}
temp_rmse = {}
for i in alpha_ridge:
    lasso_reg = Lasso(alpha=i, normalize=True) 
    lasso_reg.fit(X_train, y_train)
    lasso_pred = lasso_reg.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, lasso_pred)
    mse = metrics.mean_squared_error(y_test, lasso_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, lasso_pred))
    temp_mae[i] = mae
    temp_mse[i] = mse
    temp_rmse[i] = rmse


# In[19]:


print(temp_mae)
print(temp_mse)
print(temp_rmse)


# In[20]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100)
forest_fit = forest_reg.fit(X_train,y_train)
forest_pred = forest_fit.predict(X_test)


# In[23]:


plt.scatter(y_test,forest_pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y by Random Forest Regression')


# In[24]:


plt.scatter(y_test,linpred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y by Linear Regression')


# In[25]:


print('Linear Regression metrics')
print('MAE:', metrics.mean_absolute_error(y_test, linpred))
print('MSE:', metrics.mean_squared_error(y_test, linpred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, linpred)))
print('')
print('Random Forest Regression metrics')
print('MAE:', metrics.mean_absolute_error(y_test, forest_pred))
print('MSE:', metrics.mean_squared_error(y_test, forest_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, forest_pred)))


# In[ ]:




