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


# In[ ]:


train=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip")
train.head()


# In[ ]:


test=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip")
test.head()


# In[ ]:


y=train["Weekly_Sales"]


# In[ ]:


train.shape


# In[ ]:


train.dtypes


# In[ ]:


train["Date"]=train["Date"].astype("datetime64")
test["Date"]=test["Date"].astype("datetime64")
train.dtypes


# In[ ]:


#train["day of week"]=train["Date"].dt.dayofweek
#test["day of week"]=test["Date"].dt.dayofweek
train["month"]=train["Date"].dt.month
test["month"]=test["Date"].dt.month
train["day"]=train["Date"].dt.day
test["day"]=test["Date"].dt.day
train["year"]=train["Date"].dt.year
test["year"]=test["Date"].dt.year


# In[ ]:


#train["day of week"].value_counts()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
A,B=plt.subplots(1,1,figsize=(10,10))
sns.boxplot(train["month"],train["Weekly_Sales"])


# In[ ]:





# In[ ]:


store=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv")
store.head()


# In[ ]:


train=pd.merge(train,store)
test=pd.merge(test,store)


# In[ ]:


train["Type"].unique()


# In[ ]:


train["Type"]=train["Type"].replace({"A":1,"B":2,"C":3})
test["Type"]=test["Type"].replace({"A":1,"B":2,"C":3})


# In[ ]:


train=train.drop(["Weekly_Sales","Date"],axis=1)
train.head()


# In[ ]:


tesst=test.drop(["Date"],axis=1)
tesst.head()


# In[ ]:





# In[ ]:


#from lightgbm import LGBMRegressor 
#lgb=LGBMRegressor()
#lgb.fit(train,y)

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf.fit(train,y)


# In[ ]:


value=rf.predict(tesst)
value


# In[ ]:


sample=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip")
sample.head()


# In[ ]:


sample["Weekly_Sales"]=value


# In[ ]:


sample.head()


# In[ ]:


sample.to_csv("20200309.csv",index=False)


# In[ ]:





# In[ ]:





# In[ ]:




