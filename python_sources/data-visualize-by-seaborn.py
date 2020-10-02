#!/usr/bin/env python
# coding: utf-8

# Data Visualization by using seaborn.

# In[ ]:


import numpy as np
import pandas as pd
import itertools
import collections
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv('../input/anime.csv',header=0)
cus_data = pd.read_csv('../input/rating.csv',header=0)


# In[ ]:


data.dtypes


# In[ ]:


cus_data.dtypes


# In[ ]:


data.describe()


# In[ ]:


cus_data.describe()


# In[ ]:


data = data.replace('Unknown',np.nan)
data = data.dropna()
data['episodes'] = data['episodes'].astype('int64')


# In[ ]:


sns.pairplot(data=data[['type','members','episodes','rating']],hue='type')


# In[ ]:


sns.boxplot(data=data,x='type',y='rating')


# In[ ]:


data['genre']=data['genre'].apply(lambda x : x.split(', '))
genre_data = itertools.chain(*data['genre'].values.tolist())
genre_counts = collections.Counter(genre_data)


# In[ ]:


genre_counts


# In[ ]:


genre = pd.DataFrame.from_dict(genre_counts,orient='index').reset_index().rename(columns={'index':'genre',0:'counts'})
genre = genre.sort_values('counts',ascending=False)


# In[ ]:


sns.barplot(y=genre['genre'],x=genre['counts'],color='skyblue')


# In[ ]:


def mapper(data,col):
    if col in data:
        return 1
    elif col not in data:
        return 0
genre_collections = pd.DataFrame([],columns=genre_counts.keys())
for col in genre_collections:
    genre_collections[col] = data['genre'].apply(mapper,args=(col,))


# In[ ]:


genre_collections.head()


# In[ ]:


sns.heatmap(genre_collections.corr())

