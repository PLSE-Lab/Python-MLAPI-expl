#!/usr/bin/env python
# coding: utf-8

# # Applying LinearRegression 

# **Trying to predict "mpg", so it's our target variable (y).**

# In[ ]:


import pandas as pd


# In[ ]:


df =pd.read_table('../input/auto-mpg-from-uci-site-directly/auto-mpg.data-original.txt', delim_whitespace=True, names=('mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','car name'))


# In[ ]:


df.info()


# *One can see __"mpg"__ column has few missing data, replacing those "nan" values with median.* 

# In[ ]:


df['mpg'].median()


# In[ ]:


df['mpg'].fillna(value=df['mpg'].median(), inplace=True)


# In[ ]:


df.info()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


ls =df.columns.tolist()

df[ls].isnull().sum()


# In[ ]:


df["horsepower"].fillna(value=df["horsepower"].median(), inplace=True)  #same for "horsepower" column replacing nan with median


# In[ ]:



sns.pairplot(data=df)


# *__"mpg"__ seems negatively correlated to __"displacement","horsepower" & "weight"__ and positively correlated to __"acceleration"__.*

# In[ ]:


df_feature = df[["displacement","horsepower","weight","acceleration"]]


# In[ ]:


df_target = df["mpg"]


# In[ ]:


plt.rcParams['figure.figsize']=(10,10)
plt.hist(df_target,bins=15)


# *Our target column is not properly nomally distributed, there is slight skewness. One way to handle this is __log transformation__*

# In[ ]:


import numpy as np

df_target_norm = np.log1p(df_target)


# In[ ]:


plt.rcParams['figure.figsize']=(10,10)
plt.hist(df_target_norm,bins=15)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# _Trying to see the difference in results with and without log transformation._

# In[ ]:


x_train_n,x_test_n,y_train_n,y_test_n = train_test_split(df_feature,df_target_norm, test_size=0.33, random_state=65)


x_train,x_test,y_train,y_test = train_test_split(df_feature,df_target, test_size=0.33, random_state=65)


# In[ ]:


lr = LinearRegression().fit(x_train,y_train)
lr_n = LinearRegression().fit(x_train_n,y_train_n)


# In[ ]:


lr.score(x_train,y_train)


# In[ ]:


lr.score(x_test,y_test)


# In[ ]:


lr_n.score(x_train_n,y_train_n)


# In[ ]:


lr_n.score(x_test_n,y_test_n)


# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


y_pred = lr.predict(x_test)
y_pred_n = lr_n.predict(x_test_n)


# In[ ]:


from math import sqrt
rmse = sqrt(mean_squared_error(y_true=y_test,y_pred=y_pred))
rmse_n = sqrt(mean_squared_error(y_true=y_test_n,y_pred=y_pred_n))

print("\n non log-transformed- %f \n log-transformed- %f" %(rmse,rmse_n))


# ### Non Log-transformed

# In[ ]:


plt.rcParams['figure.figsize']=(10,10)
plt.hist(y_pred-y_test)


# In[ ]:


plt.rcParams['figure.figsize']=(8,8)
plt.hlines(y=0,xmin=0,xmax=5000)
plt.scatter(x_test["weight"],y_pred-y_test)


# ### Log Transformed-

# In[ ]:


plt.rcParams['figure.figsize']=(10,10)
plt.hist(y_pred_n-y_test_n)


# In[ ]:


plt.rcParams['figure.figsize']=(8,8)
plt.hlines(y=0,xmin=0,xmax=5000)
plt.scatter(x_test["weight"],y_pred_n-y_test_n)


# In[ ]:





# *__Experimenting by log-transforming the feature columns also:__*

# In[ ]:


df_features_norm = np.log1p(df_feature)


# In[ ]:


x2_train,x2_test,y2_train,y2_test = train_test_split(df_features_norm,df_target_norm, test_size=0.33, random_state=65)


# In[ ]:





# In[ ]:


lr2 = LinearRegression().fit(x2_train,y2_train)


# In[ ]:


lr2.score(x2_train,y2_train)


# In[ ]:


lr2.score(x2_test,y2_test)


# In[ ]:


lr2_y_pred = lr2.predict(x2_test)


# In[ ]:


rmse_lr2 = sqrt(mean_squared_error(y_true=y2_test,y_pred=lr2_y_pred))

print("\n %f" %(rmse_lr2))


# In[ ]:


plt.rcParams['figure.figsize']=(10,10)
plt.hist(lr2_y_pred-y2_test)


# In[ ]:


plt.rcParams['figure.figsize']=(8,8)
plt.hlines(y=0,xmin=6,xmax=10)
plt.scatter(x2_test["weight"],lr2_y_pred-y2_test)


# In[ ]:




