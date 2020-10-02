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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import sys


# Removed a record which had wrong number of data, like 11 fields instead of 10

# So there are 13714 rows and 10 columns...

# In[ ]:


df = pd.read_csv('../input/books.csv', error_bad_lines = False)
df.shape


# In[ ]:


df.head(5)


# In[ ]:


plt.figure(figsize=(10,7))
author_count=df['authors'].value_counts()[:10]
sns.barplot(x=author_count,y=author_count.index,palette='rocket')
plt.title("Top 10 authors with most number of books")
plt.xlabel("Number of Books Written")


# **Looks like Agatha Cristie and Stephen King have written most number of books**

# In[ ]:


highest_rated = df.sort_values('ratings_count', ascending = False).head(10).set_index('title')
plt.figure(figsize=(15,10))
sns.barplot(highest_rated['ratings_count'], highest_rated.index, palette='deep')


# **Wow, So many Harry Potters in the race**

# * I want to know which  books are lowest rated...

# In[ ]:


lowest_rated = df.sort_values('ratings_count', ascending = True).head(10).set_index('title')
plt.figure(figsize=(5,10))
sns.barplot(lowest_rated['ratings_count'].notnull(), lowest_rated.index, palette='Set3')


# Its pointless , there are so many ..

# In[ ]:


from subprocess import check_output
from wordcloud import WordCloud
wordcloud = (WordCloud(width=1440, height=1080, relative_scaling=0.5).generate_from_frequencies(df['language_code'].value_counts()))


fig = plt.figure(1,figsize=(15, 15))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# This WordCloud is for Languages of Books Present in the GoodReads Website

# In[ ]:


df_cdb=df[df['language_code']=='fre']
df_cdb.head(5)

plt.figure(figsize=(15,10))
locs=df_cdb['authors'].value_counts()[:10]
sns.barplot(x=locs,y=locs.index,palette='Set3')
plt.title("Top 10 Authors with most number of books in French Books")
plt.xlabel("Number of Books")


# In[ ]:



df.groupby(['authors','language_code'])['average_rating'].mean().sort_values()[-50:][:-1]


# In[ ]:


most_rated = df.sort_values('# num_pages', ascending = False).head(10).set_index('title')
plt.figure(figsize=(15,10))
sns.barplot(most_rated['# num_pages'], most_rated.index, palette='deep')


# In[ ]:


#most_rated.groupby(['average_rating'])
plt.figure(figsize=(20,10))
sns.barplot(most_rated['average_rating'],most_rated['# num_pages'],  palette='deep')


# Lets find HIghest rated book with the lowest number of Pages , which is The Feynman Leacutes on Physics, WOW!
# 

# In[ ]:



dfp=df['# num_pages'] > 5
dfr=df['average_rating'] > 4.5
df[dfp & dfr].sort_values('# num_pages', ascending = True).head(10)


# Books with most number of Text Reviews Count

# In[ ]:


most_rated = df.sort_values('text_reviews_count', ascending = False).head(10).set_index('title')
plt.figure(figsize=(15,10))
sns.barplot(most_rated['text_reviews_count'], most_rated.index, palette='rocket')


# Its Twilight (The part 1 ofcourse)

# In[ ]:


range_df = df.groupby(pd.cut(df['# num_pages'], [-1,500,1000,1500,2000,3000,4000,5000,10000]))
range_df = range_df[['ratings_count']]
range_df.sum().reset_index()


# So the ratings count Reduce as the number of Pages of the Books increases.

# In[ ]:


range_df = df.groupby(pd.cut(df['average_rating'], [0,1,2,3,4,5]))
range_df = range_df[['ratings_count']]
range_df.sum().reset_index()


# So people don't generally tend to rate a book so low, or may be books with such low ratings have very few readers !
# 

# In[ ]:




