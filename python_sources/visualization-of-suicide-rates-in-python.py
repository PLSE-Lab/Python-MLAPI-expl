#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


data = pd.read_csv('../input/master.csv')
data.head()


# In[36]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data.columns


# In[5]:


data.shape


# In[6]:


data.isnull().sum()


# In[7]:


data.isnull().mean()


# Since 'HDI for Year' has almost 70% null values we will drop this column. Also 'country-year' column does not contain anything useful as we have separate country and year columns. So we will drop it too.

# In[8]:


data = data.drop(['HDI for year', 'country-year'], axis = 1)


# In[9]:


data.columns


# 'HDI for Year' and 'country-year' are removed. Now we need to rename columns to something more convenient.

# In[10]:


data = data.rename(columns = {'country':'Country', 'year':'Year', 'sex':'Sex', 'age':'Age', 'suicides_no':'Suicides', 'population':'Population', 
                             'suicides/100k pop':'SuicideRate', ' gdp_for_year ($) ':'GDP',
                              'gdp_per_capita ($)': 'GDPperCapita', 'generation': 'Generation'})


# Checking column names is necessary gdp_for_year ($) has unnecessary spaces in the beginning and end.

# In[11]:


data.columns


# In[12]:


data.describe()


# In[13]:


data.info()


# Checking Object Variables

# In[14]:


data['Country'].unique()


# In[15]:


#Number of Countries in Dataset
len(data['Country'].unique())


# In[16]:


data['Sex'].unique()


# In[17]:


data['Age'].unique()


# In[18]:


data['GDP'].unique()


# In[19]:


data['Generation'].unique()


# In[20]:


data.corr()


# In[21]:


sns.heatmap(data.corr())


# ### Analyzing Suicides by Year

# In[22]:


data_year = data.groupby('Year')


# In[23]:


plt.figure(figsize= (15,5))
plt.bar(x = data_year['Suicides'].sum().keys(), height = data_year['Suicides'].sum())
plt.show()


# In[24]:


plt.figure(figsize= (15,5))
plt.bar(x = data_year['SuicideRate'].mean().keys(), height = data_year['SuicideRate'].mean())
plt.show()


# In[ ]:





# ### Analyzing Suicides by Country

# In[25]:


data_country = data.groupby('Country')


# In[26]:


plt.figure(figsize=(20,10))
height = 100*data_country['Suicides'].sum()/data['Suicides'].sum()
x = data_country['Suicides'].sum().keys()
plt.bar(x = x, height= height)
plt.xticks(rotation='vertical')
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='red', ha='center', rotation='vertical', va='bottom')
plt.show()


# There are some countries which have a exceptionally high number of suicides.

# In[27]:


height['Japan'] + height['Russian Federation'] + height['United States']


# 'Japan', 'Russian Federation' and 'United States' account for 45% of all suicides all over the world.

# In[37]:


plt.figure(figsize=(20,10))
height = data_country['SuicideRate'].mean()
x = data_country['SuicideRate'].mean().keys()
plt.bar(x = x, height= height)
plt.xticks(rotation='vertical')
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='red', ha='center',rotation='vertical',va='bottom')
plt.show()


# ### Analyzing Suicides by Gender

# In[31]:


data_gender = data.groupby('Sex')


# In[38]:


plt.figure(figsize=(10,4))
height = 100*data_gender['Suicides'].sum()/data['Suicides'].sum()
x =  data_gender['Suicides'].sum().keys()
plt.bar(x = x, height = height)
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='red', ha='center', fontweight='bold')
plt.show()


# In[33]:


plt.figure(figsize=(10,4))
height = data_gender['SuicideRate'].mean()
x =  data_gender['SuicideRate'].mean().keys()
plt.bar(x = x, height = height)
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')
plt.show()


# ### Analyzing Suicides by Age-Group

# In[34]:


data_age = data.groupby('Age')


# In[35]:


plt.figure(figsize=(10,4))
height = 100*data_age['Suicides'].sum()/data['Suicides'].sum()
x =  data_age['Suicides'].sum().keys()
plt.bar(x = x, height = height)
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')
plt.show()


# In[40]:


plt.figure(figsize=(10,4))
height = data_age['SuicideRate'].mean()
x =  data_age['SuicideRate'].mean().keys()
plt.bar(x = x, height = height)
for i,v in enumerate(height):
    plt.text(i, v, " "+str(round(v,2)), color='blue', ha='center', fontweight='bold')
plt.show()

