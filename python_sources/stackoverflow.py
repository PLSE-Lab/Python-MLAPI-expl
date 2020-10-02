#!/usr/bin/env python
# coding: utf-8

# In[8]:


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


# In[4]:


#importing necessary libraries
import pandas as  pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


df = pd.read_csv('../input/survey_results_public.csv')


# In[10]:


df.describe()


# In[11]:


df1 = df.Country.value_counts().nlargest(25)
df2 = pd.DataFrame({'country':df1.index, 'count':df1.values})
f, ax = plt.subplots(figsize=(25, 10))
sns.barplot(y='country',x='count', data=df2).set_title('Top 25 countries who participated')


# In[12]:


df3 = df[['Respondent','Country','Gender']]
df4 = df3.loc[df['Gender']=='Female']
df5 = df4.Country.value_counts().nlargest(25)
df6 = pd.DataFrame({'Country':df5.index, 'Count':df5.values})
df6.head()
f, ax = plt.subplots(figsize=(25, 10))
sns.barplot(y='Country',x='Count', data=df6).set_title('Top 25 countries with Female respondents')


# In[13]:


df3 = df[['Respondent','Country','OpenSource']]
df4 = df3.loc[df['OpenSource']=='Yes']
df5 = df4.Country.value_counts().nlargest(10)
df6 = pd.DataFrame({'Country':df5.index, 'Count':df5.values})
df6.head()
f, ax = plt.subplots(figsize=(25, 10))
sns.barplot(x='Country',y='Count', data=df6).set_title('Top 10 countries contributing to open source')


# In[14]:


import itertools as it, pandas as pd
df1 = df['DevType'].str.split(';',expand=True)
df2=df1.fillna('')
df3 = df2.stack().value_counts().nlargest(25)
df4 = pd.DataFrame({'DevType':df3.index, 'Count':df3.values})
df5 = df4.iloc[1:]
f, ax = plt.subplots(figsize=(25, 10))
sns.barplot(y='DevType',x='Count', data=df5).set_title('Top 25 Develpoer Types')


# In[ ]:




