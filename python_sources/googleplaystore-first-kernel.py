#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/googleplaystore.csv')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.describe()


# In[ ]:


df.dropna(subset=["Content Rating"],inplace=True)


# In[ ]:


df.drop(["Current Ver","Android Ver"],axis=1,inplace=True)


# In[ ]:


df.info()


# In[ ]:


df.Category.unique()


# In[ ]:


df.Type.unique()


# In[ ]:


df.Type.value_counts()


# In[ ]:


df.Type.fillna('Free',inplace=True)


# In[ ]:


df.Rating.value_counts().head(4)


# In[ ]:


df.Rating.value_counts().head(10).plot(kind='barh')


# In[ ]:


df.groupby("Category").Rating.mean()


# In[ ]:


df.groupby("Category").Rating.median()


# In[ ]:


df.Rating = df.Rating.fillna(df.groupby("Category").Rating.transform('median'))


# In[ ]:


df.info()


# I will change Install object dtype to float dtype

# In[ ]:


df.Price.value_counts().head(3)


# In[ ]:


df.Price = df.Price.str.replace("$","")
df.Price.value_counts().head(3)


# In[ ]:


df.Size.value_counts().head(3)


# In[ ]:


df.Size.unique()


# In[ ]:


df.Size = df.Size.str.replace("Varies with device", 'NaN')
df.Size = df.Size.str.replace("M", "e6")
df.Size = df.Size.str.replace("k", "e3")


# In[ ]:


df.Size = df.Size.astype(float)


# In[ ]:


df.dropna(subset=['Size'],inplace=True)


# In[ ]:


df.info()


# Look at the Installs Column. This Column has , and + string. So I change this string to blink

# In[ ]:


df.Installs.value_counts().head(3)


# In[ ]:


df.Installs.unique()


# In[ ]:


df.Installs = df.Installs.str.replace("+","")
df.Installs = df.Installs.str.replace(",","")


# In[ ]:


df.Installs.value_counts()


# In[ ]:


df.Installs = df.Installs.astype(float)


# In[ ]:


df.sample(10)


# In[ ]:


df.Installs.value_counts()


# In[ ]:


df["Last Updated"] = pd.to_datetime(df["Last Updated"])
df.Category = df.Category.astype("category")
df.Reviews = df.Reviews.astype(float)
df.Type = df.Type.astype("category")
df.Price = df.Price.astype(float)
df["Content Rating"] = df["Content Rating"].astype("category")
df.Genres = df.Genres.astype("category")
df.info()


# ***EDA**

# In[ ]:


df.columns = df.columns.str.replace(" ","_")


# In[ ]:


sns.distplot(df.Rating,kde=False, rug=True)


# In[ ]:


fig = plt.figure(figsize=(18,18))
sns.countplot(x='Rating',hue='Type',data=df,dodge=True)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(12,12))
sns.scatterplot(data=df,x="Reviews",y="Rating",hue="Type")


# In[ ]:


f, ax = plt.subplots(figsize=(12,12))
sns.scatterplot(data=df,x="Reviews",y="Rating",hue="Type")
plt.xlim(0,10**6)

plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(12,12))
sns.scatterplot(data=df,x="Reviews",y="Rating",hue="Type",size='Installs')

plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(12,12))
plt.scatter('Reviews','Rating',data=df)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(10,10))
g =sns.regplot(data=df,x="Reviews",y="Installs",fit_reg=False,x_jitter=.1)
g.set(xscale='log',yscale='log')
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(10,10))
sns.countplot(x=df.loc[(df.Type == "Paid") & (df.Price<5),"Price"])
plt.xticks(rotation=60)


# In[ ]:


f, ax = plt.subplots(figsize=(10,10))
sns.countplot(x=df.loc[(df.Type == "Paid") & (df.Price >5) & (df.Price<10) ,"Price"])
plt.xticks(rotation=60)


# In[ ]:


f, ax = plt.subplots(figsize=(10,10))
g =sns.regplot(x=df.loc[df.Type == "Paid","Price"],y=df.loc[df.Type == "Paid","Rating"],fit_reg=False)

plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(10,10))
g =sns.regplot(x=df.loc[(df.Type == "Paid")&(df.Price<50),"Price"],y=df.loc[(df.Type == "Paid")&(df.Price<50),"Rating"],fit_reg=False)

plt.show()


# In[ ]:


df.Category.value_counts().head(4)


# In[ ]:


df.Category.value_counts().head(4)


# ## Relationship between Rating and Size

# In[ ]:


sns.regplot(data=df, x="Size", y="Rating",scatter_kws={'alpha':0.3})


# In[ ]:


sns.countplot(x="Content_Rating",data=df)


# In[ ]:


sns.countplot(x=df.Rating.loc[df.Content_Rating == "Everyone"],data=df)
plt.xticks(rotation=60)


# In[ ]:


sns.catplot(x="Rating",y="Category", data=df)

