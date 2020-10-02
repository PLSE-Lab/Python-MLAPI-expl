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


df = pd.read_csv('../input/books.csv',error_bad_lines=False)
df.head(3)


# In[ ]:


df[df.text_reviews_count>10][['title','average_rating','text_reviews_count']].sort_values(by=['average_rating','text_reviews_count'], ascending=[0,0]).head(20)


# From looking at the top rated books with at least ten written reviews I can spot quite a few on Calvin and Hobbes. 
# 
# Lets check whether all Calvin and Hobbes books are highly rated or if there's poorly rated ones at the bottom as well.

# In[ ]:


df[(df.text_reviews_count>10)&(df["title"].str.contains(r'Calvin and Hobbes'))][['title','authors','average_rating','text_reviews_count']].sort_values(by=['average_rating','text_reviews_count'], ascending=[0,0])


# Ah, appears to be a clear favourite amongst the GoodReads' readers.

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


sns.distplot(df.average_rating)


# In[ ]:


print("Skewness: %f" % df['average_rating'].skew())
print("Kurtosis: %f" % df['average_rating'].kurt())


# Look at that negative skew and upright kurtosis score. These books are for the most part only being scored between three and five.

# In[ ]:


df = df[df.text_reviews_count>9]


# In[ ]:


sns.distplot(df.average_rating)


# In[ ]:


df.describe()


# I've got rid of any books with less than ten text reviews, as the less reviews the more weight is given to each critic. I'd rather base my choices on the opinions of the masses, than a book that might have been rated five stars by an author's personal fanclub.

# In[ ]:


threshold = sum(df.average_rating)/len(df.average_rating)
df['Score'] = [1 if i > threshold else 0 for i in df.average_rating]
df.head(3)


# I've calculated a threshold which lays at about 3.75. If a book is rated over that it's a good score, below and it's bad.
# 
# Let's take a look at which authors have the best (and worst) ratio of good to bad books.

# In[ ]:


author_df = df[['authors','Score']].groupby('authors').sum()
author_df['book_count']=df['authors'].groupby(df.authors).count()
author_df['ratio'] = author_df['Score']/author_df['book_count']
author_df[author_df.ratio == 1].sort_values('book_count', ascending=False).head(10)


# Every book by these authors that has been reviewed on GoodReads is considered good when compared to all the books reviewed on the site.
# 
# Basically get yourself a book by one of these authors and you're going to have a good time.

# In[ ]:





# In[ ]:




