#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')


# In[ ]:


data.head()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


plt.figure(figsize = (40,20))
sns.countplot(x='Country',data=data)


# ## China has been severy maligned with this and so much so it has traveleled as far as Brazil and Finland which is surprising and amazing and shocking at the same time.

# Lets see the number of confirmed cases in each country.

# In[ ]:


cc= data.groupby('Country')['Confirmed'].sum().reset_index(drop = False).sort_values(by = 'Confirmed', ascending = False)


# In[ ]:


cc


# In[ ]:


plt.figure(figsize = (40,20))
sns.countplot(x='Confirmed',data=cc)


# In[ ]:


plt.figure(figsize=(35,17))
sns.heatmap(data.drop('Sno', axis = 1).corr(),annot=True,cmap='viridis')


# In[ ]:


data


# ###  Lets see which states are suffering the most

# In[ ]:


state = data.groupby('Province/State')['Confirmed'].sum().reset_index(drop = False).sort_values(by = 'Confirmed', ascending = False)


# In[ ]:


state


# In[ ]:


plt.figure(figsize=(35,17))
sns.scatterplot(x='Province/State',y='Confirmed',data=data)


# * Lets see the death by Countries which is expected to be 100 percent in China

# In[ ]:


death = data.groupby('Country')['Deaths'].sum().reset_index(drop = False).sort_values(by = 'Deaths', ascending = False)


# In[ ]:


death


# Lets see which states have seen most casualty

# In[ ]:


death_state = data.groupby('Province/State')['Deaths'].sum().reset_index(drop = False).sort_values(by = 'Deaths', ascending = False)


# In[ ]:


death_state


# I believe Hubei is somewhere very close to Wuhan or is part of Wuhan in some sense.

# In[ ]:


plt.figure(figsize=(35,17))
sns.countplot(x='Deaths',data=death_state,hue='Province/State')


# ##### Lets see recovered

# In[ ]:


data.Recovered.unique()


# In[ ]:


plt.figure(figsize=(35,17))
data.Recovered.plot()


# In[ ]:


rec_state = data.groupby('Province/State')['Recovered'].sum().reset_index(drop = False).sort_values(by = 'Recovered', ascending = False)


# In[ ]:


rec_state


# In[ ]:


plt.figure(figsize=(35,17))
sns.countplot(x='Recovered',data=rec_state,hue='Province/State')


# # More coming soon. Upvote if you liked it.

# In[ ]:




