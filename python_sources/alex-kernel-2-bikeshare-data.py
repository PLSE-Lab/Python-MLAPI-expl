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


df = pd.read_csv("../input/bike_share.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.describe().T


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# In[ ]:


df.duplicated().sum()


# In[ ]:


df[df.duplicated()]


# In[ ]:


df = df.drop_duplicates()


# In[ ]:


df.shape


# In[ ]:


df.windspeed.plot(kind="box")


# In[ ]:


df.casual.plot(kind="box")


# In[ ]:


df.registered.plot(kind="box")


# In[ ]:


df.corr()


# In[ ]:


df.corr()['count']


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,3))
corr=df.corr()
sns.heatmap(corr,annot=True)


# In[ ]:


df.windspeed.median()


# In[ ]:


x  = df.drop(columns=["count"])
x.columns


# In[ ]:


y = df[['count']]
y.columns


# In[ ]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=1)


# In[ ]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_x, train_y)


# In[ ]:


predict_train = model.predict(train_x)
predict_test = model.predict(test_x)


# In[ ]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[ ]:


train_MSE = mean_squared_error(train_y, predict_train)
test_MSE = mean_squared_error(test_y, predict_test)


# In[ ]:


print("train_MSE: ",train_MSE)
print("test_MSE: ",test_MSE)


# In[ ]:


train_MAE = mean_absolute_error(train_y, predict_train)
test_MAE= mean_absolute_error(test_y, predict_test)


# In[ ]:


print("train_MAE: ",train_MAE)
print("test_MAE: ",test_MAE)


# In[ ]:


train_RMSE = np.sqrt(train_MSE)
test_RMSE = np.sqrt(test_MSE)


# In[ ]:


train_MAPE = np.mean(np.abs(train_y, predict_train))
test_MAPE = np.mean(np.abs(test_y, predict_test))


# In[ ]:


print("train_MAPE: ", train_MAPE)
print("test_MAPE: ", test_MAPE)


# In[ ]:


r2_train = r2_score(train_y,predict_train)
r2_test = r2_score(test_y,predict_test)


# In[ ]:


print("r2_train: ", r2_train)
print("r2_test: ", r2_test)

