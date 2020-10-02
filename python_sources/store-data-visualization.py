#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Warnings
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10, 7)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train.head()


# In[ ]:


# set datatime
train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])


# In[ ]:


# create datetime variable
train['tyear'] = train['date'].dt.year
train['tmonth'] = train['date'].dt.month
train['tday'] = train['date'].dt.day


test['tyear'] = test['date'].dt.year
test['tmonth'] = test['date'].dt.month
test['tday'] = test['date'].dt.day


# In[ ]:


g=sns.FacetGrid(train,col="store", col_order=[1,2,3,4,5,6,7,8,9,10],col_wrap=2,size=5)
g.map(sns.barplot,"tyear","sales")


# Every year there is increase in sales each store

# In[ ]:


train[['sales','store']].groupby(["store"]).mean().plot.bar(color='c')
plt.show()


# Store 2 & 8 Have higest sales

# In[ ]:


train[['sales','tmonth']].groupby(["tmonth"]).mean().plot.bar(color='g')
plt.show()


# June,July & Aug have high sales

# In[ ]:


train[['sales','tyear']].groupby(["tyear"]).mean().plot.bar(color='lightblue')
plt.show()


# Every year there is increase sales

# In[ ]:


train[['sales','tday']].groupby(["tday"]).mean().plot.bar(color='lightgreen')
plt.show()


# **Store 1  Analysis**

# In[ ]:


data_1=train.loc[train['store'] == 1]


# In[ ]:


data_1.head()


# In[ ]:


data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar()
plt.show()


# In[ ]:


print("Top 5 selling item in store 1")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())


# In[ ]:


print("lest 5 selling item in store 1")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())


# **Store 2  Analysis**

# In[ ]:


data_1=train.loc[train['store'] == 2]


# In[ ]:


data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar(color='lightblue')
plt.show()


# In[ ]:


print("Top 5 selling item in store 2")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())


# In[ ]:


print("lest 5 selling item in store 2")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())


# **Store 3  Analysis**

# In[ ]:


data_1=train.loc[train['store'] == 3]


# In[ ]:


data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar(color='g')
plt.show()


# In[ ]:


print("Top 5 selling item in store 3")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())


# In[ ]:


print("lest 5 selling item in store 3")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())


# **Store 4  Analysis**

# In[ ]:


data_1=train.loc[train['store'] == 4]


# In[ ]:


data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar(color='y')
plt.show()


# In[ ]:


print("Top 5 selling item in store 4")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())


# In[ ]:


print("lest 5 selling item in store 4")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())


# **Store 5  Analysis**

# In[ ]:


data_1=train.loc[train['store'] == 1]


# In[ ]:


data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar(color='lightblue')
plt.show()


# In[ ]:


print("Top 5 selling item in store 5")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())


# In[ ]:


print("lest 5 selling item in store 5")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())


# **Store 6  Analysis**

# In[ ]:


data_1=train.loc[train['store'] == 6]


# In[ ]:


data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar()
plt.show()


# In[ ]:


print("Top 5 selling item in store 6")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())


# In[ ]:


print("lest 5 selling item in store 6")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())


# **Store 7  Analysis**

# In[ ]:


data_1=train.loc[train['store'] == 7]


# In[ ]:


data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar(color='y')
plt.show()


# In[ ]:


print("Top 5 selling item in store 7")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())


# In[ ]:


print("lest 5 selling item in store 7")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())


# **Store 8  Analysis**

# In[ ]:


data_1=train.loc[train['store'] == 8]


# In[ ]:


data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar(color='lightblue')
plt.show()


# In[ ]:


print("Top 5 selling item in store 8")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())


# In[ ]:


print("lest 5 selling item in store 8")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())


# **Store 9  Analysis**

# In[ ]:


data_1=train.loc[train['store'] == 9]


# In[ ]:


data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar(color='b')
plt.show()


# In[ ]:


print("Top 5 selling item in store 9")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())


# In[ ]:


print("lest 5 selling item in store 9")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())


# **Store 10  Analysis**

# In[ ]:


data_1=train.loc[train['store'] == 10]


# In[ ]:


data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).plot.bar()
plt.show()


# In[ ]:


print("Top 5 selling item in store 10")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).head())


# In[ ]:


print("lest 5 selling item in store 10")
print(data_1[['sales','item']].groupby(["item"]).mean().sort_values(by='sales',ascending=False).tail())


# In[ ]:





# In[ ]:




