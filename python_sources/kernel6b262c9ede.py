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





# In[ ]:


import pandas as pd
a = pd.read_csv("../input/train.csv")


# In[ ]:


a.head()


# In[ ]:


a['genres']


# In[ ]:


df=list()
for i in range(10):
  st=a['genres'][i].split()[3]
  st = st[1:-3]
  df.append(st)
print(df)


# In[ ]:


a.isnull().sum()


# In[ ]:



a.head()


# In[ ]:


ax=["belongs_to_collection","homepage","production_countries","tagline","Keywords"]


# In[ ]:


a


# In[ ]:


a.drop(ax,axis=1,inplace = True) 
  


# In[ ]:


a


# In[ ]:


a["genres"]


# In[ ]:


df=list()
for i in range(471):
  st=a['genres'][i].split()[3]
  st = st[1:-3]
  df.append(st)
print(df)


# In[ ]:


a.iloc[470]


# In[ ]:


a.iloc[100]


# In[ ]:


a = a.fillna(a['spoken_languages'].value_counts(),inplace=True)


# In[ ]:


a.iloc[470]


# In[ ]:


df=list()
for i in range(471):
  st=a['genres'][i].split()[3]
  st = st[1:-3]
  df.append(st)
print(df)


# In[ ]:



imp = SimpleImputer(strategy="most_frequent")
print(imp.fit_transform(a['spoken_languages'].reshape(-1,1)))


# In[ ]:


for each

