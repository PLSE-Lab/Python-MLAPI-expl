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


df = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.drop('id',axis=1,inplace=True)


# In[ ]:


df['date'] = [i.split('-')[0] for i in df['date']]


# In[ ]:


df['date'].unique()


# In[ ]:


df['manner_of_death'].unique()


# In[ ]:


df['armed'].value_counts()


# In[ ]:


armed_stats = df.groupby('armed')['armed'].agg('count').sort_values(ascending = False)


# In[ ]:


df['armed'] = df['armed'].apply(lambda x:'other' if x in armed_stats[armed_stats<=25] else x)


# In[ ]:


df['armed'].fillna('other',inplace=True)


# In[ ]:


df['armed'].unique()


# In[ ]:


df['armed'] = np.where((df.armed == 'undetermined'),'other',df.armed)


# In[ ]:


df['armed'].unique()


# In[ ]:


df['armed'] = np.where((df.armed == 'unknown weapon'),'other',df.armed)


# In[ ]:


df['armed'].unique()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df.age.min()


# In[ ]:


df.age.max()


# In[ ]:


df.age.median()


# In[ ]:


df.age.fillna(df.age.median(),inplace=True)


# In[ ]:


import seaborn as sns


# In[ ]:


sns.distplot(df['age'])


# In[ ]:


df.gender.isnull().sum()


# In[ ]:


df.gender.value_counts()


# In[ ]:


df.gender.fillna('M',inplace=True)


# In[ ]:


sns.countplot(x='gender',data=df)


# In[ ]:


sns.countplot(x='gender',hue='signs_of_mental_illness',data=df)


# In[ ]:


df.race.unique()


# In[ ]:


df.race.value_counts()


# In[ ]:


df.race.fillna('O',inplace=True)


# In[ ]:


df.race.isnull().sum()


# In[ ]:


sns.countplot(x='race',hue='manner_of_death',data=df)


# In[ ]:


sns.countplot(x='race',hue='signs_of_mental_illness',data=df)


# In[ ]:


df['state'].unique()


# In[ ]:


plt.figure(figsize=(20,20))
sns.countplot(x='state',data=df)


# In[ ]:


graph= sns.countplot(x='flee',hue='manner_of_death',data=df)


# In[ ]:





# In[ ]:




