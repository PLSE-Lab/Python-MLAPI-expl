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


apps = pd.read_csv("../input/googleplaystore.csv")
apps.head()


# ## There are plenty of duplicate rows

# In[ ]:


apps[apps['App'] == 'Subway Surfers']


# ## Drop duplicates rows 

# In[ ]:


apps.drop_duplicates(subset='App',inplace = True)


# In[ ]:


apps[apps['App'] == 'Subway Surfers']


# In[ ]:


len(apps)


# In[ ]:


apps.columns


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


len(apps['Category'].unique())


# In[ ]:


apps['Category'].unique()


# ## Finding categorywise total apps
# 
# 

# In[ ]:


apps_series = apps['Category'].value_counts()


# In[ ]:


apps_series.index[:-1]


# In[ ]:


apps_df = pd.DataFrame({
    'Category' : apps_series.index[:-1],
    'Total_apps' : apps_series.values[:-1]
})
apps_df.head()


# In[ ]:


apps_df.set_index('Category' , inplace=True)


# In[ ]:


apps_df.sort_values(by='Total_apps' , inplace=True , ascending= False)
apps_df.head()


# In[ ]:


apps_df.plot.barh(figsize=(10,20))
plt.xlabel('Number of apps')
plt.ylabel('Category')
plt.title("Categorywise apps on playstore")


# ## Most popular Genres apps people are installed

# In[ ]:


apps['Installs'].unique()


# In[ ]:


def set_installs(installs):
    if installs == 'Free':
        return int('-2')
    elif installs == '0':
        return int('-1')
    elif installs == '0+':
        return int('0')
    return int(installs[:-1].replace(',' , '0'))


# In[ ]:


installs = apps['Installs'].apply(set_installs)
installs.head()


# In[ ]:


most_install_apps = pd.DataFrame({
    'App':apps['App'],
    'Category' :apps['Category'],
    'Genres':apps['Genres'],
    'Installs' : installs,
    'Rating' : apps['Rating'],
    'Type' : apps['Type']
})
most_install_apps.head()


# In[ ]:


most_install_apps.sort_values(by='Installs' , ascending=False , inplace=True)
most_install_apps.head()


# In[ ]:


Genreswise = most_install_apps.groupby('Genres')
Genreswise


# In[ ]:


Genreswise.get_group('Communication').head()


# In[ ]:


most_demanding_genre = most_install_apps.pivot_table(index='Genres')


# In[ ]:


most_demanding_genre.head()


# In[ ]:


most_install_apps[most_install_apps['Genres'] == 'Action'].mean()


# In[ ]:


most_demanding_genre.sort_values(by='Installs' , ascending=False , inplace=True)
most_demanding_genre.head()


# In[ ]:


most_demanding_genre['Rating'].fillna(0 , inplace=True)


# In[ ]:


most_demanding_genre.iloc[:5].plot.barh(figsize=(20,10))


# ## Average rating of each Genres

# In[ ]:


del most_demanding_genre['Installs']
most_demanding_genre


# In[ ]:


most_demanding_genre.plot.barh(figsize=(10,30) , color='green')


# In[ ]:


import seaborn as sns


# In[ ]:


apps.Type.unique()


# In[ ]:


import seaborn as sns


# ## Most applications types on playstore

# In[ ]:


sns.countplot(apps['Type'])


# In[ ]:


type_of_app = most_install_apps.pivot_table(index = 'Type')


# In[ ]:


type_of_app


# ## Most highest rating  free apps

# In[ ]:


free_app = most_install_apps[most_install_apps['Type']=='Free']


# In[ ]:


free_app[free_app['Rating'] == free_app['Rating'].max()].head()


# ## Most highest rating  paid apps

# In[ ]:


paid_app = most_install_apps[most_install_apps['Type']=='Paid']
paid_app[paid_app['Rating'] == paid_app['Rating'].max()].head()


# # Most Installed Free Apps

# In[ ]:


free_app[free_app['Installs'] == free_app['Installs'].max()].head()


# ## Most Installed paid Apps

# In[ ]:


paid_app[paid_app['Installs'] == paid_app['Installs'].max()].head()


# In[ ]:


apps['Content Rating'].unique()


# In[ ]:




