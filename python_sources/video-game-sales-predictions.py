#!/usr/bin/env python
# coding: utf-8

# # Predicting The Sales of Video Games [WIP]
# ![Arcade](https://images.unsplash.com/photo-1513528473392-f3fffb1b31a9?ixlib=rb-0.3.5&ixid=eyJhcHBfaWQiOjEyMDd9&s=44845c69fe4febae451d4133ccb7518b&auto=format&fit=crop&w=750&q=80)
# 

# The purpose of this experiment is to practice Univariate and Multivariate EDA. 

# In[ ]:


#Libraries/Dependices

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)

import os
print(os.listdir("../input"))


# In[ ]:


PATH = "../input/Video_Games_Sales_as_at_22_Dec_2016.csv"


# In[ ]:


df_Games = pd.read_csv("../input/Video_Games_Sales_as_at_22_Dec_2016.csv", index_col=0)


# In[ ]:


df_Games.head(30)


# In[ ]:


df_Games.info()


# In[ ]:


df_Games.describe()


# ## EDA Time!

# Global Sales

# In[ ]:


sns.distplot(df_Games['Global_Sales'], kde=False)


# Year of Release

# In[ ]:


sns.distplot(df_Games['Year_of_Release'].dropna())


# Critic Score

# In[ ]:


sns.distplot(df_Games['Critic_Score'].dropna())


# Platforms

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(df_Games['Platform'].sort_index())


# Genres

# In[ ]:


plt.figure(figsize=(14,8))
sns.countplot(df_Games['Genre'].sort_index())


# Publishers

# Developers

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(df_Games['Developer'].dropna(), order = df_Games['Developer'].value_counts().iloc[:40].index)
plt.xticks(rotation=90);


# Lets try and get an understanding of what variables have good correlations each other.

# In[ ]:





# 

# Lets try and find the genre with the most sales [A BIT LATER AS ITS Multivariate analysis]

# In[ ]:


sales_genres = df_Games[['Genre', 'Global_Sales']]
sales_genres.head()


# 
