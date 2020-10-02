#!/usr/bin/env python
# coding: utf-8

# In[74]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/Suicides Rates Overview 1985 to 2016/master.csv"))

# Any results you write to the current directory are saved as output.


# In[80]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Hello and Welcome all to my very first kernel. 
# Topic : Suicide Rates Overview 1985 to 2016

# In[ ]:


dataset = pd.read_csv('../input/master.csv')


# In[ ]:


dataset.head()


# In[ ]:


dataset.tail()


# Step 1: Information related to dataset

# In[ ]:


dataset.info()


# In[ ]:


dataset.describe()


# In[ ]:


dataset.shape


# Step 2: Data Cleaning

# In[ ]:


dataset.columns


# In[ ]:


del dataset['country-year']


# In[ ]:


del dataset['HDI for year']


# In[ ]:


dataset.rename(columns={'gdp_for_year ($) ' : 'gdp_for_year'},inplace = True)


# In[ ]:


dataset.rename(columns={'gdp_per_capita ($)':'gdp_per_capita'},inplace = True)


# In[ ]:





# Step 3: Data visualisation
# 1. Data correlation to each other
# 2. Basic plots of some relevant and important columns
# 3. comparison of the information.

# In[ ]:


dataset.corr()


# In[ ]:


plt.figure(figsize= (10,7))
sns.heatmap(dataset.corr(),annot=True)


# * ** population vs Years**

# In[ ]:


years = dataset.year.unique()
years = sorted(years)


# In[75]:


population = []
for year in years:
    population.append([dataset[dataset['year']==year]['population'].sum()])

plt.plot(years, population,'-o')
plt.ylabel('Population -->')
plt.xlabel('Years --> ')
plt.show()


# * **No. of suicides vs population**

# In[76]:


suicides = []
for year in years:
    suicides.append([dataset[dataset['year']==year]['suicides_no'].sum()])
plt.plot(years, suicides,'-o')
plt.ylabel('Suicides -->')
plt.xlabel('Years --> ')
plt.show()


# Dividing on the basis of gender

# In[77]:


plt.figure(figsize=(10,7))
sns.barplot(x='age',y='suicides_no',hue='sex',data=dataset)
plt.show()


# Dividing on the basis of Generation

# In[78]:


generation = pd.unique(dataset['generation'])
gen_pos = np.arange(len(generation))

g_suic=[dataset[dataset['generation']==gen]['suicides_no'].sum() for gen in generation]
    
plt.barh(generation,g_suic)
plt.yticks(gen_pos,generation)


# In[79]:


plt.figure(figsize=(10,25))
sns.countplot(y='country',data=dataset,alpha=0.7)


# Thanks. Will be updated whenever some interesting insights be found.

# > **Upvote if you like**

# In[ ]:




