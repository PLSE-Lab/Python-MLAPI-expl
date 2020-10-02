#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Import Data
df = pd.read_csv('../input/tmdb_5000_movies.csv')


# In[ ]:


# First 5 Rows
df.head(5)


# Before start tidying copy the dataframe for not losing any information

# In[ ]:


df_tidy = df.copy()


# In[ ]:


df_tidy.head(5)


# In[ ]:


import time


# In[ ]:


df_tidy.info()


# genres, keywords, production_companies, production_countries and spoken_languages coloumns need to be simplified.
# * They have code before informaion (ex: [{"name": "Ingenious Film Partners", "id": 289... ))
# * Every unique value has 6 character long. We divide every line to 6 to find how many items we have. 

# In[ ]:


def tidying_row(dataframe):
    start = time.clock()
    lenght=len(dataframe)
    for row in range(0,lenght):
        new_list = []
        iter_time = len(dataframe.iloc[row].split('\"'))//6
        stry = dataframe.iloc[row].split('\"')
        for i in range(1,iter_time+1):
            new_list.append(stry[(i*6)-1])
        dataframe.iloc[row] = new_list
    print (time.clock() - start)
tidying_row(df_tidy.genres.head(4803))


# It took 1.261097 seconds to tidy genres column. "df_tidy.genres.head(4803)" in this line without ".head(4803)" it took longer.
# * We need to apply function to keywords, production_companies, production_countries and spoken_languages columns

# In[ ]:


tidying_row(df_tidy.keywords.head(4803))
tidying_row(df_tidy.production_companies.head(4803))
tidying_row(df_tidy.production_countries.head(4803))
tidying_row(df_tidy.spoken_languages.head(4803))


# Print out the execution time. They all look fine to me.
# * Check the new tidied database

# In[ ]:


df_tidy.head(5)


# In[ ]:


df_tidy_high_vote = df_tidy[df_tidy.vote_average > 8]
df_tidy_high_vote.corr()


# Filtered the list to include high-rated movies (more than 8). There is a correlation between Budget and Revenue 

# In[ ]:


plt.figure(figsize=(8, 8), dpi=80)
plt.scatter(df_tidy_high_vote.budget, df_tidy_high_vote.revenue, alpha=0.5)
plt.title("Budget vs Revenue")
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.show()


# The graph above shows that revenue is increase with the budget.

# In[ ]:


df_tidy_high_vote.describe()


# In[ ]:


df_tidy_high_vote.info()


# homepage and tagline columns have Null object.

# In[ ]:


df_tidy_high_vote.homepage.fillna('UNKNOWN',inplace=True)
df_tidy_high_vote.tagline.fillna('UNKNOWN',inplace=True)
df_tidy_high_vote.info()


# As homepage and tagline columns are not so important for our study we fill Null values with string 'UNKNOWN'

# In[ ]:


df_tidy_high_vote.boxplot(column='revenue', figsize=(8,8))
plt.show()


# According to descriptive statistics there is some Outlier in movie revenues. 

# In[ ]:


alist = ['Science Fiction']
df_tidy_high_vote[df_tidy_high_vote.genres.apply(lambda x :set(alist).issubset(x))]
df_tidy_high_vote['SciFi'] = np.where(df_tidy_high_vote.genres.apply(lambda x :set(alist).issubset(x)), 'Yes', 'No')


# Add a column according to Movie is SciFi or not

# In[ ]:


df_tidy_high_vote.head()


# In[ ]:


df_tidy_high_vote.boxplot(column='revenue', by='SciFi', figsize=(8,8))
plt.show()


# * Non SciFi movie's revenues have a wide seperation from 0 to above 1 billion
# * On the other hand all SciFi films have near revenue and it is more then Half Billion.
# * In conclusion if you want to have a higher revenue SciFi films are the best option for you.
