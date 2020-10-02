#!/usr/bin/env python
# coding: utf-8

# In[5]:


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


# In[15]:


#Import libraries Untuk Memasukan Data

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


#Mengatur Kolom Sebagai Index
df=pd.read_csv("../input/Mall_Customers.csv",index_col=0)


# In[44]:


#Periksa Data #NB DF.head bisa dikosongkan bisa juga di isi seberapa banyak data yang mau di tampilkan
df.head(5)


# In[46]:


#untuk memeriksa info data
df.info()


# In[12]:


df.describe()


# In[29]:


df.plot.scatter(x='Annual Income (k$)', y = 'Spending Score (1-100)')


# In[47]:


sns.lmplot(x='Age', y = 'Spending Score (1-100)', hue='Gender', data=df)


# In[35]:


sns.pairplot(df, palette='inferno')


# In[38]:


X1 = df.loc[:,['Age', 'Spending Score (1-100)']]


# In[49]:


#untuk menampilakan id costumer umur dan score head bisa di isi bebas sesuai data yang mau di tampilkan
X1.head(3)

