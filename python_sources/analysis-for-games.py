#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

import seaborn as sns
sns.set_style("dark")
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# > # A study for Games data from year 1980 to 2020 including:
# - Basic insighs and correlation between varibales
# - Visulization for trends in games and platform in games histroy*

# In[ ]:


df=pd.read_csv("../input/videogamesales/vgsales.csv")
df.columns=["rank","name","platform","year","genre","publisher","na_sales","eu_sales","jp_sales","other_sales","global_sales"]
# first 5 rows to have an overview
df.head()


# #### In this data set there are 11 columns. Their names and data types as follows:
# - Rank - Ranking of overall sales, integer
# - Name - The games name, object
# - Platform - Platform of the games 
# - Year - Year of the game's release, float
# - Genre - Genre of the game ,object
# - Publisher - Publisher of the game
# - NA_Sales - Sales in North America
# - EU_Sales - Sales in Europe
# - JP_Sales - Sales in Japan 
# - Other_Sales - Sales in the rest of the world
# - Global_Sales - Total worldwide sales

# In[ ]:


df.describe()


# In[ ]:


df.corr(method ='pearson') 


# In[ ]:


sns.pairplot(df)


# In[ ]:


year_data = df['year']
print("Max Year Value: ", year_data.min())


# In[ ]:


#Filling the missing values
df.isnull().sum()


# In[ ]:


best_genre = df[['genre', 'global_sales']]
best_genre.groupby('genre').sum()


# In[ ]:


# pie plot to illustrate the diffrence
plt.figure(figsize=(12,8))
colors = sns.color_palette("muted")
ax = sns.barplot(data = best_genre, y = best_genre.genre, x= best_genre.global_sales)
ax.set_xlabel(xlabel='Global Sales', fontsize=16)
ax.set_ylabel(ylabel='Genre', fontsize=16)
ax.set_title('Comparison between Global Sales for every Genre', fontsize=18)


# ### Best Globa Sales year

# In[ ]:


best_year = df[['year', 'global_sales']]
best_year.groupby('global_sales')
plt.figure(figsize=(12,8))

ax = sns.barplot(data = best_year, x=best_year.year, y = best_year.global_sales)
ax.set_xticklabels(labels='Years', fontsize=12, rotation=50)
ax.set_xlabel(xlabel='Years', fontsize=16)
ax.set_ylabel(ylabel='Revenue Per Game in Millions', fontsize=16)
ax.set_title(label='Best Revenue Per Game by Year in $ Millions', fontsize=20)


# In[ ]:


plat_pop = pd.crosstab(df.platform,df.genre)
plat_pop_total = plat_pop.sum(axis=1).sort_values(ascending= False)
plt.figure(figsize=(10, 10))
ax = sns.barplot(y = plat_pop_total.index, x = plat_pop_total.values)
ax.set_xlabel(xlabel ='Platform', fontsize= 15 )
ax.set_ylabel(ylabel ='number of games', fontsize= 15 )
ax.set_title(label='Number of games in each platform', fontsize=20)
plt.show()


# In[ ]:




