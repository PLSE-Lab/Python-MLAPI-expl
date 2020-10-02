#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# > # Read The Data

# In[ ]:


df_app = pd.read_csv('../input/googleplaystore.csv')
df_user = pd.read_csv('../input/googleplaystore_user_reviews.csv')


# In[ ]:


df_app.head()


# In[ ]:


df_user.head()


# In[ ]:


df_app.info()


# In[ ]:


df_user.info()


# > # Preprocess Data

# ### Handle empty data

# In[ ]:


df_app.isnull().any()


# In[ ]:


df_app.dropna(inplace=True)


# In[ ]:


df_app.count()


# In[ ]:


df_user.isnull().any()


# In[ ]:


sns.heatmap(df_user.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# In[ ]:





# > # Data exploration

# ### For the data of googleplaystore.csv

# In[ ]:


df_app.columns


# In[ ]:


plt.subplots(figsize = (22, 30))
plt.title("The app counts of different category")
sns.countplot(y = 'Category', order=df_app['Category'].value_counts().index, data = df_app)


# In[ ]:


# Tryto find the top 10 popular, which Reviews is 
#plt.subplots(figsize = (10,8))
# plt.title('Top 10 highest rate app')
df_app['Reviews'] = pd.to_numeric(df_app['Reviews'])

df_app[df_app['Reviews']> 100].sort_values(by=['Rating'], ascending=False).head(20)


# In[ ]:


df_app.info()


# > # Predicit the rating(Machine Learning)

# In[ ]:




