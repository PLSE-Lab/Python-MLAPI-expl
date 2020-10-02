#!/usr/bin/env python
# coding: utf-8

# **                          High Revenue Movie from 2006 - 2016**

# My first kaggle submission. I tried to prove that movies are generating huge revenue because of technology(sci -fi, fantasy), huge cast and  **big banner**.
# 
# I felt : 
# 
# **Sci-fi, fantasy are bringing more crowd to the theatre**.
# 
# than the others (where people just want to go for online streaming in their couch).

# In[91]:


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


# Reading the csv file

# In[92]:


movies = pd.read_csv("../input/IMDB-Movie-Data.csv")
movies.head()


# In[93]:


# information about number of rows and columns
movies.shape


# In[94]:


#check the min ,max, count of movies
movies.describe()


# In[95]:


# checking for null values
movies.isnull().any()


# In[96]:


# dropping null value columns to avoid errors
movies.dropna(inplace = True)

#checking again for null values and got False for all columns
movies.isnull().any()


# In[97]:


high_revenue = movies[movies['Revenue (Millions)'] > 550.00]
high_revenue.head()

high_revenue.insert(11,'Box office (Billions)', [2.06, 1.51, 1.67, 2.78])
high_revenue_sorted = high_revenue.sort_values(by = 'Year')
high_revenue_sorted


# In[98]:


import matplotlib.pyplot as plt


# In[100]:


# Plotting year vs revenue
year = high_revenue_sorted['Year'].values
revenue = high_revenue_sorted['Revenue (Millions)'].values

plt.plot(year, revenue)
plt.xlabel("Year")
plt.ylabel("Revenue (Millions)")
plt.show()


# In[101]:


# Plotting the highest revenue and annotating the high revenue movie
fig,ax = plt.subplots()
ax.annotate(high_revenue_sorted['Title'].iloc[0],
            xy = (2009,760), xycoords = 'data',
            xytext = (2009,780), textcoords = 'data',
            arrowprops = dict(arrowstyle = '->',
                             connectionstyle = 'arc3'),
           )

ax.annotate(high_revenue_sorted['Title'].iloc[1],
            xy = (2012,623), xycoords = 'data',
            xytext = (2012,650), textcoords = 'data',
            arrowprops = dict(arrowstyle = '->',
                             connectionstyle = 'arc3'),
           )

ax.annotate(high_revenue_sorted['Title'].iloc[2],
            xy = (2015,936), xycoords = 'data',
            xytext = (2015,950), textcoords = 'data',
            arrowprops = dict(arrowstyle = '->',
                             connectionstyle = 'arc3'),
           )

ax.annotate(high_revenue_sorted['Title'].iloc[3],
            xy = (2015,652), xycoords = 'data',
            xytext = (2015,630), textcoords = 'data',
            arrowprops = dict(arrowstyle = '->',
                             connectionstyle = 'arc3'),
           )


year = high_revenue_sorted['Year'].values
revenue = high_revenue_sorted['Revenue (Millions)'].values

plt.plot(year, revenue)
plt.xlabel("Year")
plt.ylabel("Revenue (Millions)")
#plt.axis([2009,2015,0,950])
plt.show()


# From the above graph and exploring movies from 2006-2016 
# 
# I found that:
# A movie can generate high revenue if
# * the movie is action, sci-fi or fantasy or all three ( use of lot of technology in making)
# * Big marketing and big banner ( Walt disney(50% in above graph), Universal, 20th century Fox)
# 
# Avatar was a stunning leap in 3D technology and effects, making for an incredible in-theater experience that was unlike anything people had seen in many years. --Quora (What makes the movie Avatar so successful?)
# 
# Star Wars(The Force awakens) Disney backed the film with extensive marketing campaigns and was released on late December (you get two weeks of weekdays that play like weekends).

# In[ ]:




