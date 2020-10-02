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


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df=pd.read_csv('/kaggle/input/world-happiness-report/2020.csv')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


print('total number of countries are {}'.format(df['Country name'].nunique()))


# In[ ]:


df['Country name'].head(10)


# In[ ]:


plt.figure(figsize=(12,6))
sns.barplot(x=df['Country name'].head(10),y='Ladder score',data=df,palette='viridis')
plt.title('A plot of the top ten happiest countries to live versus their ladder score')
plt.tight_layout()


# In[ ]:


plt.figure(figsize=(14,8))
sns.barplot(x=df['Country name'].tail(10),y='Ladder score',data=df,palette='Set1')
plt.tight_layout()


# In[ ]:


df['Regional indicator'].unique()


# In[ ]:


df.groupby('Regional indicator').sum().sort_values(by='Ladder score',ascending=False)


# In[ ]:


df[df['Healthy life expectancy']==df['Healthy life expectancy'].max()]


# In[ ]:


df[df['Healthy life expectancy']==df['Healthy life expectancy'].min()]


# In[ ]:


import cufflinks as cf
cf.go_offline()


# In[ ]:


df.iplot(kind='scatter',x='Country name',y='Healthy life expectancy')


# In[ ]:


df.groupby('Country name')['Logged GDP per capita'].sum().sort_values(ascending=False).head(10)


# In[ ]:


df.groupby('Country name')['Social support'].sum().sort_values(ascending=False)


# In[ ]:


plt.figure(figsize=(10,6))
sns.scatterplot(x='Social support',y='Ladder score',data=df)


# In[ ]:


df.groupby('Country name')['Perceptions of corruption'].sum().sort_values(ascending=True)


# In[ ]:


df.set_index('Regional indicator').head()


# In[ ]:


plt.figure(figsize=(21,6))
sns.barplot(x=df['Regional indicator'],y='Ladder score',data=df)
plt.tight_layout()


# In[ ]:




