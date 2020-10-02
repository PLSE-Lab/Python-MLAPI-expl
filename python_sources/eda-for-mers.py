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


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[ ]:


country_count_latest=pd.read_csv('/kaggle/input/mers-outbreak-dataset-20122019/country_count_latest.csv')
weekly_clean=pd.read_csv('/kaggle/input/mers-outbreak-dataset-20122019/weekly_clean.csv')


# In[ ]:


weekly_clean


# In[ ]:


country_count_latest


# In[ ]:


country_count_latest=country_count_latest.sort_values(by='Confirmed', ascending=False)


# In[ ]:


country_count_latest=country_count_latest.reset_index()


# In[ ]:


country_count_latest


# To Find The Total Number of Cases Worldwide.

# In[ ]:


country_count_latest['Confirmed'].sum()


# Number of Cases in each Country

# In[ ]:


fig = plt.figure(figsize=(20,10))

ax = fig.add_axes([0,0,1,1])
ax.bar(country_count_latest['Country'], country_count_latest['Confirmed'])
plt.xlabel('Country')
plt.ylabel('Count')
plt.show()


# Number of Cases in Top 5 countries

# In[ ]:


fig = plt.figure(figsize=(20,10))

ax = fig.add_axes([0,0,1,1])
ax.bar(country_count_latest['Country'].head(5), country_count_latest['Confirmed'].head(5))
plt.xlabel('Country')
plt.ylabel('Count')
plt.show()


# In[ ]:


fig = plt.figure(figsize=(30,30))
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
langs = ['C', 'C++', 'Java', 'Python', 'PHP']
students = [23,17,35,29,12]
ax.pie(country_count_latest['Confirmed'], labels = country_count_latest['Country'],autopct='%1.2f%%')
plt.show()


# Total Cases

# In[ ]:


weekly_clean['New Cases'].sum()


# Number of New Cases for every week. 

# In[ ]:


weekly_clean.groupby(['Year', 'Week'])['New Cases'].sum()


# Number of Cases in Each Region

# In[ ]:


weekly_clean.groupby(['Region'])['New Cases'].sum()


# In[ ]:


weekly_cum_overall=pd.DataFrame(weekly_clean.groupby(['Year', 'Week'])['New Cases'].sum())


# In[ ]:


weekly_cum_overall=weekly_cum_overall.reset_index()


# In[ ]:


weekly_cum_overall


# How Number of new cases vary with time every week

# In[ ]:


weekly_cum_overall['New Cases'].plot(figsize=(20,10))
plt.xlabel('Time')
plt.ylabel('New Cases')


# How Number of new cases vary with time every week for each region

# In[ ]:


regions=weekly_clean['Region'].unique()


# In[ ]:


for i in regions:
    print(i)
    weekly_clean[weekly_clean['Region']==i]['New Cases'].plot(figsize=(20,10))
    plt.xlabel('Time')
    plt.ylabel('New Cases')
    plt.show()


# Overlapped Graphs For comparision

# In[ ]:


#overlapped
for i in regions:
    weekly_clean[weekly_clean['Region']==i]['New Cases'].plot(figsize=(20,10))
plt.xlabel('Time')
plt.ylabel('New Cases')
plt.show()

