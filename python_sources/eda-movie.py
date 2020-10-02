#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA # Principal Component Analysis module
from sklearn.cluster import KMeans # KMeans clustering 
import matplotlib.pyplot as plt # Python defacto plotting library
import seaborn as sns # More snazzy plotting library
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


movie=pd.read_csv('../input/movie_metadata.csv') 


# In[ ]:


movie.columns


# In[ ]:


movie.head()


# In[ ]:


movie['profit']=(((movie['gross'].values)-(movie['budget'].values))/movie['gross'].values)*100


# In[ ]:


f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(movie.corr(),linewidths=1,vmax=1.0, square=True)


# In[ ]:


# lets visulize percent of profit from 1980 to as on date
g = sns.jointplot(x="title_year", y="profit",kind='scatter',size=10,ylim = [0,110],xlim=[1980,2020],data=movie)


# In[ ]:


h = sns.jointplot(x="imdb_score", y="profit",kind='reg',size=10,ylim = [0,110],data=movie)


# In[ ]:


g = sns.pairplot(movie,hue='content_rating')

