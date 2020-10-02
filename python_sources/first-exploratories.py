#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

from unidecode import unidecode


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Util functions

# In[33]:


def bar(acumm_data):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    ax = sns.barplot(x=acumm_data.index, y=acumm_data.values, palette='tab20b', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.show()
def stackedbar(data, rotation=90):
    ax = data.loc[:,data.columns].plot.bar(stacked=True, figsize=(10,7))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    plt.tight_layout()
    plt.show()  
def lower_undescore_unicode(lst):
    return [unidecode(re.sub(' ','_',x.lower())) for x in lst]


# # Read data and first preparation

# In[3]:


df = pd.read_csv('../input/IHMStefanini_industrial_safety_and_health_database.csv', parse_dates=['Data'])


# ### First look

# In[4]:


df.info()


# In[5]:


df.head()


# In[6]:


df.columns = lower_undescore_unicode(df.columns)


# ### Replace roman numbers

# In[7]:


df.accident_level.unique()


# In[8]:


df.potential_accident_level.unique()


# In[9]:


convert_roman = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6}


# In[10]:


df.accident_level = df.accident_level.replace(convert_roman)
df.potential_accident_level = df.potential_accident_level.replace(convert_roman)


# In[11]:


df.head()


# In[12]:


df.info()


# # First exploratory analysis

# > ### Genre proportions

# In[13]:


accum_genre = df.genre.value_counts()
bar(accum_genre)


# In[14]:


print(f'Percentage of Male: {round(100*accum_genre["Male"]/(accum_genre["Male"]+accum_genre["Female"]),2)}%')
print(f'Percentage of Female: {round(100*accum_genre["Female"]/(accum_genre["Male"]+accum_genre["Female"]),2)}%')


# ### Sector proportions

# In[15]:


bar(df.industry_sector.value_counts())


# ### Risk proportions

# In[16]:


# remove Others and print bar
bar(df.risco_critico[~df['risco_critico'].isin(['Others'])].value_counts())


# ### Employee proportions

# In[17]:


bar(df.employee_ou_terceiro.value_counts())


# ### Accident vs. Potential Accidents

# In[18]:



freq_matrix = pd.crosstab(df.accident_level, df.potential_accident_level)
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.heatmap(freq_matrix, square=True, cmap='Purples', ax=ax, annot=True)


# ### Frequency time series by week

# In[22]:


df.set_index('data')['local'].resample('W').count().plot()
plt.show()


# ### Accidents by week day

# In[20]:


df['weekday'] = df.data.dt.dayofweek+1


# In[21]:


df.weekday.value_counts().sort_index().plot()
plt.show()


# ### Stacked accident Level by genre

# In[34]:


freq_matrix = pd.crosstab(df.genre, df.accident_level)
stackedbar(freq_matrix.T)


# ### Staked accidente level by industry sector

# In[45]:


freq_matrix = pd.crosstab(df.industry_sector, df.accident_level)
stackedbar(freq_matrix.T)


# ### Heatmap Accident level/Potencial accident level by local

# In[44]:


freq_matrix1 = pd.crosstab(df.local, df.accident_level)
freq_matrix2 = pd.crosstab(df.local, df.potential_accident_level)
fig, ax = plt.subplots(figsize=(20,10),ncols=2)
ax[0] = sns.heatmap(freq_matrix1, square=True, cmap='Purples', annot=True, ax=ax[0])
ax[1] = sns.heatmap(freq_matrix2, square=True, cmap='Purples', annot=True, ax=ax[1])
ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=0)
ax[1].set_yticklabels(ax[1].get_yticklabels(), rotation=0)

plt.show()


# In[ ]:


df.genre.unique()


# In[ ]:


df.genre = df.genre.replace({'Male': 1, 'Female': 0})


# # to be continued...

# In[ ]:




