#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


path="/kaggle/input/avocado-prices/avocado.csv"
df=pd.read_csv(path)
df.head()#for first 5 dataset


# In[ ]:


df.tail() # for last 5 datasets


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


#averge of price
df["AveragePrice"].mean()


# In[ ]:


df["type"].value_counts()


# In[ ]:


df["region"].value_counts()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df.hist(bins=50,figsize=(25,15))
plt.show()


# In[ ]:


#max average price of new york
df[df["region"]=="NewYork"]["AveragePrice"].max()


# In[ ]:


# averge price of Newyork
df[df["region"]=="NewYork"]["AveragePrice"].mean()


# In[ ]:


# Max average price from all regions
df[df["AveragePrice"]==df["AveragePrice"]].max()


# In[ ]:


# Min average price from all regions
df[df["AveragePrice"]==df["AveragePrice"]].min()


# In[ ]:


df[df["region"]=="SanFrancisco"].max()


# In[ ]:


plt.figure(figsize=(15,9))
plt.scatter(x=df["region"],y=df["AveragePrice"])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# define category wise 
plt.figure(figsize=(19,10))
sns.scatterplot(x=df["region"],y=df["AveragePrice"],hue=df["type"])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


# Average Price per year
sns.jointplot(x=df["year"], y=df["AveragePrice"],data=df,kind="hex", color="#4CB396")
plt.show()


# In[ ]:





# In[ ]:




