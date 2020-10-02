#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


path = "../input/1000-cameras-dataset/camera_dataset.csv"
df = pd.read_csv(path)


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# In[ ]:


df.describe


# In[ ]:


df.ndim


# In[ ]:


df.columns


# In[ ]:


df.count


# In[ ]:


df.isnull().sum()


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.drop(['Model'],axis=1,inplace=True)


# In[ ]:


df.drop(['Release date'],axis=1,inplace=True)


# In[ ]:


df.shape


# In[ ]:


df['Macro focus range'].fillna(df['Macro focus range'].mean())


# In[ ]:


df['Storage included'].fillna(df['Storage included'].mean())


# In[ ]:


df['Weight (inc. batteries)'].fillna(df['Weight (inc. batteries)'].mean())


# In[ ]:


df['Dimensions'].fillna(df['Dimensions'].mean())


# In[ ]:


df['Price'].fillna(df['Price'].mean())


# In[ ]:


train , test = train_test_split(df,test_size=0.3)


# In[ ]:


print(train.shape)


# In[ ]:


print(test.shape)


# In[ ]:


train_x = train.iloc[:,0:3]; train_y = train.iloc[:,3]
test_x  = test.iloc[:,0:3];  test_y = test.iloc[:,3]


# In[ ]:


print(train_x)


# In[ ]:


print(test_x)


# In[ ]:


print(train_x.shape)


# In[ ]:


train_y.shape


# In[ ]:


test_x.shape


# In[ ]:


test_y.shape


# In[ ]:


train_x.head()


# In[ ]:


train_y.head()


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


train.dtypes


# In[ ]:


fit = sm.OLS(train_y,train_x).fit()


# In[ ]:


#Predction
pred = fit.predict(test_x)


# In[ ]:


pred


# In[ ]:


#Actual
actual = list(test_y.head(103))


# In[ ]:


actual


# In[ ]:


predicted = np.round(np.array(list(pred.head(103))),2)


# In[ ]:


predicted


# In[ ]:


type(predicted)


# In[ ]:


#Actual vs Predicted
df_results = pd.DataFrame({'actual':actual, 'predicted':predicted})


# In[ ]:


df_results


# # To check accuracy

# In[ ]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, pred))  


# In[ ]:


print('Mean Squared Error:', metrics.mean_squared_error(test_y,pred))


# In[ ]:


print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pred)))  

