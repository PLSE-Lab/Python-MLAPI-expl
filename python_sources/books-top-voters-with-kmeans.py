#!/usr/bin/env python
# coding: utf-8

# * groupby, aggregations (count, sum, average)
# * multiple bar charts
# * kmeans clustering

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[2]:


def wavg(group, avg_name, weight_name):
    """ http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
    In rare instance, we may not have weights, so just return the mean. Customize this if your business case
    should return otherwise.
    """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()


# <h2>The datasource</h2>
# joining the two csv tables to have:
# * title
# * user
# * rating

# In[3]:


df_books = pd.read_csv('../input/books.csv')[['book_id', 'goodreads_title']]
df_rat = pd.read_csv('../input/ratings.csv')
df = pd.merge(df_books, df_rat)[['goodreads_title', 'user_id', 'rating']]
print(len(df.groupby('goodreads_title')), "books")
print(len(df.groupby('user_id')), "users")
df.head()


# <h2>The top rated books</h2>
# calculating the averages for each book and sorting

# In[4]:


df_rating_stats_by_book = df[['goodreads_title', 'rating']].groupby(['goodreads_title']).agg(['mean', 'count']).sort_values([('rating', 'mean')], ascending=False)
df_rating_stats_by_book.head()


# In[5]:


df_best_avg = df[df.goodreads_title == 'ESV Study Bible']
df_best_avg['rating'].value_counts().plot.bar()
plt.title('rating frequency for the top rated "ESV Study Bible"')
plt.xlabel('rating')
plt.ylabel('frequency')
plt.show()


# <h2>Distribution of ratings</h2>
# 

# In[6]:


df_rating_stats_by_book['rating'].head()
df_rating_stats_by_book['rating'].groupby(pd.cut(df_rating_stats_by_book['rating']['mean'], bins=np.linspace(1,5,17))).sum()['count'].plot.bar()
plt.ylabel('frequency')
plt.title('distribution of the rating average')
plt.show()


# the distribution looks very similar to Gaussian, except it is biased toward the good ratings

# <h2>Clustering the users based on their ratings</h2>
# for each user specifying the frequency of the different ratings

# In[7]:


df_user_rating = df[['user_id', 'rating']]
df_user_rating_freq = df_user_rating.groupby(['user_id', 'rating'])['user_id'].agg(['count'])
df_user_rating_freq.head()


# In[8]:


df_rating_pivot = df_user_rating_freq.pivot_table(index='user_id', columns='rating', values='count', fill_value=0)
#df_rating_pivot['rating_sum'] = df_rating_pivot.sum(axis=1)
df_rating_pivot.head()


# 5 feature vectors, represeting the rating categories (1-5)

# In[9]:


kmeans = KMeans(n_clusters=5).fit(df_rating_pivot.as_matrix())
df_rating_pivot['cluster'] = kmeans.labels_


# In[10]:


df_rating_pivot.groupby('cluster').sum().plot.bar()
plt.title('rating distribution in each cluster')
plt.show()


# In[11]:


df_rating_clusters = df_rating_pivot.reset_index().melt(id_vars=["user_id", "cluster"], var_name="rating", value_name="freq")
df_rating_clusters = df_rating_clusters[df_rating_clusters.freq != 0]
df_rating_clusters.groupby('cluster').apply(wavg, 'rating', 'freq').plot.bar()
plt.title('rating averages through clusters')
plt.ylabel('rating average')
plt.show()


# TODO:
# * do the clustering with the column sum as a new column -> it will bias the clustering <- both categorical and numerical variables (kmodes)
# * removing top voters and recalculating the rating distribution

# In[ ]:




