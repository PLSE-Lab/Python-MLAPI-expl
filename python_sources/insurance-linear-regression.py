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


df = pd.read_csv("../input/insurance.csv")
df.head()


# In[ ]:


df.index


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# In[ ]:


df.corr()


# In[ ]:


df.region.unique()


# In[ ]:


df.smoker.unique()


# In[ ]:


df.replace({"yes":"1","no":"0"},inplace=True)


# In[ ]:


df.replace({"southwest":"0","southeast":"1","northwest":"2","northeast":"3"},inplace=True)


# In[ ]:


df.sex.unique()


# In[ ]:


df.replace({"female":"1","male":"0"},inplace=True)


# In[ ]:


df["sex"] = pd.to_numeric(df["sex"], errors='coerce')
df["smoker"] = pd.to_numeric(df["smoker"], errors='coerce')
df["region"] = pd.to_numeric(df["region"], errors='coerce')
df.info()


# In[ ]:


df['bmi'] = df['bmi'].astype('Int64')
df['expenses'] = df['expenses'].astype('Int64')
df.info()


# In[ ]:


df.corr()


# In[ ]:


x= df.drop(["expenses","sex","children","region"],axis=1)
x.head()


# In[ ]:


y=df["expenses"]
y.head()


# In[ ]:


import seaborn as sns
for i in x.columns:
    sns.pairplot(data=df,x_vars=i,y_vars="expenses")


# In[ ]:


g = sns.PairGrid(df, y_vars=["expenses"], x_vars=["smoker", "age","bmi"])
g.map(sns.regplot)


# In[ ]:


#x= x.drop("smoker",axis=1)
x= x.drop("age",axis=1)
#x= x.drop("bmi",axis=1)
x.head()


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=123)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics
model = LinearRegression()
model.fit(x_train,y_train)  # Providing the training values to find model's intercept and slope


# In[ ]:


model.intercept_


# In[ ]:


model.coef_


# In[ ]:


y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)


# In[ ]:


train_aberror=metrics.mean_absolute_error(y_train,y_train_pred)
test_aberror=metrics.mean_absolute_error(y_test,y_test_pred)

train_sqerror=metrics.mean_squared_error(y_train,y_train_pred)
test_sqerror=metrics.mean_squared_error(y_test,y_test_pred)

train_sqlogerror=metrics.mean_squared_log_error(y_train,y_train_pred)
test_sqlogerror=metrics.mean_squared_log_error(y_test,y_test_pred)

train_r2Score=metrics.r2_score(y_train,y_train_pred)
test_r2Score=metrics.r2_score(y_test,y_test_pred)

print("mean_absolute_error train",train_aberror)
print("mean_absolute_error test",test_aberror)

print("mean_squared_error train",train_sqerror)
print("mean_squared_error test",test_sqerror)

print("mean_squared_log_error train",train_sqlogerror)
print("mean_squared_log_error test",test_sqlogerror)

print("r2_score train",train_r2Score)
print("r2_score test",test_r2Score)

train_rootsqerror=np.sqrt(train_sqerror)
test_rootsqerror=np.sqrt(test_sqerror)

print("squared root of mean_squared_error train",train_rootsqerror)
print("squared root of mean_squared_error test",test_rootsqerror)


# When i tried to reduce the root MSE, i tried below combination and for if i select X value as smoker and age, error value is comparatively less. Find below the error values:
# 
# smoker & age
# RMSE train 6488.319164066241
# RMSE test 6160.250014019673
# 
# age & bmi
# RMSE train 11277.753117861359
# RMSE test 11638.742215538936
# 
# smoker & bmi
# RMSE train 7182.545065097446
# RMSE test 6844.708034413751
# 
# smoker
# RMSE train 7570.46395328078
# RMSE test 7219.5676185822595
# 
# age
# RMSE train 11520.87774088801
# RMSE 11635.268757059392
# 
# bmi
# RMSE train 11849.131409547814
# RMSE test 11938.480901377052
