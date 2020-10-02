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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/suicides-in-india/Suicides in India 2001-2012.csv")
df


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df["Total"].describe()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.isna().sum()


# In[ ]:


df["State"].value_counts()


# In[ ]:


a = df.loc[df["State"]=="Karnataka"].reset_index()


# In[ ]:


# Series Way of conversion 
s = a.groupby(["Year","Gender"])["Total"].sum()
s


# In[ ]:


# Series Way of conversion 
s = a.groupby(["Year","Gender"])[["Total"]].sum()
s


# In[ ]:


type(s)


# In[ ]:


# Plotting some graphs for Exploratory Data Analysis
df


# In[ ]:


# Plotting Simple Bar Chart
sns.barplot(x="Year",y="Total",data=df,ci=0,saturation=0.5,palette="gist_earth")


# In[ ]:


from scipy import stats
sns.distplot(df["Year"],fit=stats.norm,hist=False)


# In[ ]:


from scipy import stats
sns.distplot(df["Year"],fit=stats.norm,kde=False)


# In[ ]:


## Stripplot
plt.figure(figsize=(30,10))
x = sns.stripplot(x="State",y="Total",data=df)
x.set_xticklabels(x.get_xticklabels(),rotation=30)


# In[ ]:


plt.figure(figsize=(30,10))
x = sns.boxplot(x="State",y="Total",data=df)
x.set_xticklabels(x.get_xticklabels(),rotation=30)


# In[ ]:


# indexNames = df[df['State'] == 'Total (All India)'| df['State'] == 'Total (States)'].index
# er= df.drop(indexNames)


# In[ ]:


sns.heatmap(df.corr(),annot=True)


# In[ ]:


sns.heatmap(df.describe(),annot=True,fmt="g")


# In[ ]:


sns.clustermap(df.corr(),annot=True,fmt="g")


# In[ ]:


sns.clustermap(df.describe(),annot=True,fmt="g")


# In[ ]:


# rows = df['State'] != 'Total (All India)'| df['State'] != 'Total (States)'


# In[ ]:


indexNames = df[df['State'] == 'Total (All India)'].index
er= df.drop(indexNames)


# In[ ]:


indexNames = er[er['State'] == 'Total (States)'].index
re = er.drop(indexNames)


# In[ ]:


re.reset_index()


# In[ ]:


plt.figure(figsize=(30,10))
x = sns.boxplot(x="State",y="Total",data=re)
x.set_xticklabels(x.get_xticklabels(),rotation=30)

