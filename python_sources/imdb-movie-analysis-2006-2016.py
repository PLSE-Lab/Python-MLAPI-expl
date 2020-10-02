#!/usr/bin/env python
# coding: utf-8

# IMDB Basic Movie analysis
# =========================
# This is just a basic analysis on the dataset of IMDB movies from 2006-2016. The analysis is mostly done on Revenue,Rating,DIrector of the movies.  Hope you will like it. If you like it please upvote it.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Reading the csv file and creating a dataframe object
df = pd.read_csv('../input/IMDB-Movie-Data.csv')


# In[ ]:


df.describe()


# In[ ]:


##Removing all space and brackets in the column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')


# In[ ]:


df.describe()


# In[ ]:


##Dropping NaN columns from the below columns as i will be doing analysis on these columns
df.dropna(subset=['rating', 'director','year','director'],inplace=True)


# In[ ]:


##creating a sorted array for unique number of years
np1=df.year.unique()
list1=np.sort(np1)

count=[]
##Iterating through numpy array using nditer to fetch movies more than 300 million dollars revenue each year
for i in np.nditer(list1):
    j = len(df.loc[(df.revenue_millions>300) & (df.year==i)])
    count.append(j)
    
    
print(count)


# In[ ]:


index=np.arange(len(list1))

#Defining a function to plot a bar chart for number of movies having more than $300 million revenue each year.
def plot_bar_x():
    # this is for plotting purpose
    #index = np.arange(len(label))
    plt.bar(index,count)
    plt.xlabel('Year', fontsize=10)
    plt.ylabel('No of Movies', fontsize=10)
    plt.xticks(index, list1, fontsize=10, rotation=30)
    plt.title('Number of movies having revenue greater than 300 each year')
    plt.show()
    
plot_bar_x()


# **2016** was the best year in terms of number of movies which grossed more than **$300 million.**

# In[ ]:


rating=[]
##Iterating through numpy array using nditer to fetch movies more than rating 8 each year
for i in np.nditer(list1):
    j = len(df.loc[(df.rating>8) & (df.year==i)])
    rating.append(j)
    
index=np.arange(len(list1))

#Defining a function to plot a bar chart for number of movies having more than $300 million revenue each year.
def plot_bar_x():
    # this is for plotting purpose
    #index = np.arange(len(label))
    plt.bar(index,rating,color='green')
    plt.xlabel('Year', fontsize=10)
    plt.ylabel('No of Movies', fontsize=10)
    plt.xticks(index, list1, fontsize=10, rotation=30)
    plt.title('Counting number of movies having rating greater than 8 each year')
    plt.show()
    
plot_bar_x()


# **2014** the best year in terms of number of movies greater than **rating 8**.

# In[ ]:


## Fetching top 5 highest rated movie title and its director
df.nlargest(5,'rating')[['title','director','rating','revenue_millions']].set_index('title')


# In terms of top 5 rated movies ** Christoper Nolan** has 3 entries. We have 1 entryfrom India in terms of Nitesh Tiwari for Dangal.

# In[ ]:


## Fetching top 5 highest revenue movie title and its director
df.nlargest(5,'revenue_millions')[['title','director','rating','revenue_millions']].set_index('title')


# **Star Wars : Episode VII** tops the list in terms of revenue collection with whopping $936.63 millions.

# In[ ]:


## Fetching top 5 lengthy movies title and its director
df.nlargest(5,'runtime_minutes')[['title','director','rating','runtime_minutes']].set_index('title')


# **Grindhouse** is the movie with maximum run time of more than 3 hours

# In[ ]:


##Top 10 directors based on IMDB rated movies who has directed more than 1 movie
##df.groupby('director').count()['rank'] >1 --> returns a series object where it tells whether the director has more than 1 movie 
##in the list. It is being converted to lost using tolist() method to use it as boolean indexer
df.loc[(df.groupby('director').count()['rank'] >1).tolist()].groupby('director').mean()[['rating']].nlargest(10,'rating')


# Once again **Christoper Nolan** tops the list in terms of average rating. The position is shared by **Damien Chazelle**. The analysis has been done for the directors who has directed at least 2 movies.

# In[ ]:


##Average rating of movies yearwise
df.groupby('year').mean()[['rating']].nlargest(10,'rating')


# In terms of average rating of all movies, **2007** was the best year.

# In[ ]:


##Violin plot of yearwise rating distribution using seaborn
df_yearwise_rating=df[['year','rating']]
ax = sns.violinplot(x="year", y="rating", data=df_yearwise_rating)


# The above violin plot shows the distribution of rating in a given year. 2008 sees the maximum number variance. It has seen best rated movie and poor rated movie in the same year.

# In[ ]:


##Violin plot of yearwise revenue distribution using seaborn
df_yearwise_revenue=df[['year','revenue_millions']]
ax = sns.violinplot(x="year", y="revenue_millions", data=df_yearwise_revenue)


# In the violin plot above **2015** saw the maximum variance in terms of revenue collection.
