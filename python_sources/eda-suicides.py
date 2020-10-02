#!/usr/bin/env python
# coding: utf-8

# # Explore suicide data

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# In[2]:


# Any results you write to the current directory are saved as output.

file = '../input/master.csv'

df = pd.read_csv(file)

print(df.shape)


# In[3]:


df.head()


# In[4]:


df[['country', 'year']].nunique()


# In[5]:


df.groupby('sex').agg({'suicides_no': 'sum'})


# In[6]:


df.groupby(['sex', 'generation']).agg({'suicides_no': 'sum'})


# In[8]:


d = df.groupby(['sex', 'year']).agg({'suicides/100k pop': 'mean'})

d.head()


# In[11]:


d.reset_index().head()


# ## Plots

# In[9]:


plt.style.use('ggplot')


# In[10]:


plt.figure()
df.hist(figsize=(8, 6), layout=(2, 3))
plt.tight_layout();


# In[17]:


d = df.groupby(['sex', 'year']).agg({'suicides/100k pop': 'mean'})
d.reset_index(inplace=True)
female = d.loc[d['sex']=='female']
male = d.loc[d['sex']=='male']
f, ax = plt.subplots(ncols=2, figsize=(12, 5))
female.plot(kind='bar', x='year', y='suicides/100k pop', label='female', ax=ax[0], alpha=0.5)
male.plot(kind='bar', x='year', y='suicides/100k pop', label='male', ax=ax[1], alpha=0.5)
f.tight_layout(rect=[0, 0.02, 1, 0.95])
f.suptitle('Suicides/100k Population')


# In[27]:


f, ax = plt.subplots()
df.boxplot(by='sex', column=['suicides/100k pop'], ax=ax)
f.tight_layout()
f.suptitle(None)


# In[28]:


d = df.groupby('country').agg({'suicides/100k pop': 'mean'}).sort_values(by='suicides/100k pop')
f, ax = plt.subplots(figsize=(8, 14))
d.plot(kind='barh', ax=ax)


# In[29]:


d = df[df['country']=='Thailand'].groupby('year').agg({'suicides/100k pop': 'mean'})
f, ax = plt.subplots(figsize=(8, 8))
d.plot(kind='barh', ax=ax)


# In[ ]:





# In[ ]:




