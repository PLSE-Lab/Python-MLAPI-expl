#!/usr/bin/env python
# coding: utf-8

# **import laberies**

# In[ ]:


import numpy as np 
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from tqdm import tqdm
from progressbar import ProgressBar
import re
from scipy.cluster.vq import kmeans, vq
from pylab import plot, show
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Load data

# In[ ]:


gb = pd.read_csv('../input/goodreadsbooks/books.csv', error_bad_lines = False)
print('Size of weather data frame is :',gb.shape)


# In[ ]:


#features
gb.dtypes


# In[ ]:


#Converting object datatypes to numeric
gb[['average_rating', '# num_pages', 'ratings_count', 'text_reviews_count']] = gb[['average_rating', '# num_pages', 'ratings_count', 'text_reviews_count']].convert_objects(convert_numeric=True)
gb['isbn13'] = gb['isbn13'].apply(str)
gb.dtypes


# In[ ]:


#How many null values do we have?
gb.count().sort_values()


# 
# 
# We don't have NA values!
# 

# 
# 
# Exploratory Data Analysis
# 
# 10 Authors with more books
# 

# In[ ]:


plt.figure(figsize=(16,10))
plot = sns.countplot(x = "authors", data = gb, order = gb['authors'].value_counts().iloc[:10].index, palette = "Set3")


# Top 10 most rated books

# In[ ]:


most_rated = gb.sort_values('ratings_count', ascending = False).head(10).set_index('title')
plt.figure(figsize=(15,10))
sns.barplot(most_rated['ratings_count'], most_rated.index, palette='Set3')


# 
# 
# Number of books per rating
# 

# In[ ]:


plt.figure(figsize=(10,10))
plot = sns.countplot(x = "average_rating", data = gb, order = gb['average_rating'].value_counts().iloc[:10].index, palette = "Set3")


# 
# Average rating distribution
# 

# In[ ]:


sns.distplot(gb['average_rating'], 
             kde_kws={"color": "coral", "lw": 1, "label": "KDE"}, 
             hist_kws={"histtype": "stepfilled", "linewidth": 1, "alpha": 1, "color": "skyblue"});


# 
# Number of pages distribution
# 

# In[ ]:


sns.distplot(gb['# num_pages'], 
             kde_kws={"color": "coral", "lw": 1, "label": "KDE"}, 
             hist_kws={"histtype": "stepfilled", "linewidth": 1, "alpha": 1, "color": "skyblue"});


# 
# Top 10 books with more pages
# 

# In[ ]:


most_rated = gb.sort_values('# num_pages', ascending = False).head(10).set_index('title')
plt.figure(figsize=(15,10))
sns.barplot(most_rated['# num_pages'], most_rated.index, palette='Set3')


# 
# Total number of written text reviews distribution
# 

# In[ ]:


sns.distplot(gb['text_reviews_count'], 
             kde_kws={"color": "coral", "lw": 1, "label": "KDE"}, 
             hist_kws={"histtype": "stepfilled", "linewidth": 1, "alpha": 1, "color": "skyblue"});


# 
# Top 10 books with more written text reviews
# 

# In[ ]:


most_rated = gb.sort_values('text_reviews_count', ascending = False).head(10).set_index('title')
plt.figure(figsize=(15,10))
sns.barplot(most_rated['text_reviews_count'], most_rated.index, palette='Set3')


# 
# Relationship between Rating and Text Reviews
# 

# In[ ]:


plt.figure(figsize=(15,10))
gb.dropna(0, inplace=True)
sns.set_context('paper')
ax =sns.jointplot(x="average_rating",y='text_reviews_count', kind='scatter',  data= gb[['text_reviews_count', 'average_rating']])
ax.set_axis_labels("Average Rating", "Text Review Count")
plt.show()


# 
# Relationship between Rating and Number of pages
# 

# In[ ]:


plt.figure(figsize=(15,10))
sns.set_context('paper')
ax = sns.jointplot(x="average_rating", y="# num_pages", data = gb, color = 'crimson')
ax.set_axis_labels("Average Rating", "Number of Pages")


# 
# Let's remove some outliers. We'll leave outside books with number of pages above 1500
# 

# In[ ]:


without_pages_outliers = gb[~(gb['# num_pages']>1500)]
ax = sns.jointplot(x="average_rating", y="# num_pages", data = without_pages_outliers, color = 'darkcyan')
ax.set_axis_labels("Average Rating", "Number of Pages")


# In[ ]:


gb.corr()


# In[ ]:


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(gb.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
#it can be concluded that there is probably positive correlation between text_reviews_count and ratings_count


# distribution of books for all languages

# In[ ]:


sns.set_context('paper')
plt.figure(figsize=(15,10))
ax = gb.groupby('language_code')['title'].count().plot.bar()
plt.title('Language Code')
plt.xticks(fontsize = 15)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x()-0.3, p.get_height()+100))

