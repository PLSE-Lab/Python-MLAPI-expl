#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Checking the given files**

# In[ ]:


features = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip')
features


# In[ ]:


stores = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')
stores.head()


# In[ ]:


train = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
train.head(10)


# In[ ]:


train


# In[ ]:


stores.Type.value_counts()


# **Merging train, features, stores on common columns**

# In[ ]:


df=pd.merge(train,features,on=['Store','Date','IsHoliday'],how='inner')


# In[ ]:


df.head()


# In[ ]:


df=pd.merge(df,stores,on='Store',how='inner')
df.head(10)


# **Checking for null values**

# In[ ]:


df.isna().mean()*100


# **It is observed that Markdown 1-5 have almost 65% of missing values**

# **Treating NaN values with Simple Imputer**

# In[ ]:


from sklearn.impute import SimpleImputer

markdown=pd.DataFrame(SimpleImputer().fit_transform(df[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']]),columns=['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])
df = df.drop(['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'],axis=1)
df=pd.concat([df,markdown],axis=1)


# In[ ]:


df.dtypes


# **Since date is of object datatype, I converted it to datetime type and extracting day,month,year from it separetely**

# In[ ]:


df['Date']=pd.to_datetime(df['Date'])
df['year']=df['Date'].dt.year
df['month']=df['Date'].dt.month
df['day']=df['Date'].dt.day
del df['Date']


# **Treating the object and bool columns with Label Encoding**

# In[ ]:


from sklearn.preprocessing import LabelEncoder

df['Type']=LabelEncoder().fit_transform(df['Type'])
df['IsHoliday']=LabelEncoder().fit_transform(df['IsHoliday'])
df.dtypes


# In[ ]:


df['Weekly_Sales'].plot.box()


# In[ ]:


df.columns


# **Checking for outliers**

# In[ ]:


df[['Store', 'Dept', 'Weekly_Sales', 'IsHoliday', 'Temperature',
       'Fuel_Price', 'CPI', 'Unemployment', 'Type', 'Size']].plot(kind='box',subplots=1,layout=(3,5),figsize=(14,12))


# In[ ]:


from scipy.stats import zscore 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# **Standardizing the data and splitting the data into train and test for building the model**

# In[ ]:


X = df.drop('Weekly_Sales',axis=1)
y = df['Weekly_Sales']
X_scaled = X.apply(zscore)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=.3,random_state=34)


# **Building different models to choose the best fit model with better accuracy**

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

lr = LinearRegression()
dt= DecisionTreeRegressor()
rf = RandomForestRegressor()
models = [lr,dt,rf]

for model in models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(r2_score(y_test,y_pred))


# It is observed that DecisionTreeRegressor and RandomForestRegressor have better r2 score compared to LinearRegressor model

# **Checking the feature importance on Random Forest for hyperparameter tuning to get the best model with best parameters**

# In[ ]:


(pd.DataFrame([X.columns,rf.feature_importances_],columns=['Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI',
       'Unemployment', 'Type', 'Size', 'MarkDown1', 'MarkDown2', 'MarkDown3',
       'MarkDown4', 'MarkDown5', 'year', 'month', 'day']).T).plot.bar()


# From the above bar plot, we can observed that Department have high importance compared to other features.
# Size and Store are next importance features.

#  **Dropping the least importance feature and building the model again**

# In[ ]:


pd.DataFrame([X.columns,rf.feature_importances_],columns=['Store', 'Dept', 'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI',
       'Unemployment', 'Type', 'Size', 'MarkDown1', 'MarkDown2', 'MarkDown3',
       'MarkDown4', 'MarkDown5', 'year', 'month', 'day']).T


# In[ ]:


x1 = X_scaled.drop(['IsHoliday','year','MarkDown5','MarkDown4','MarkDown1','MarkDown2'],axis=1)


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(x1,y,test_size=.3,random_state=34)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
r2_score(y_test,y_pred)


# In[ ]:




