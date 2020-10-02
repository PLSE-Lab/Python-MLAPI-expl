#!/usr/bin/env python
# coding: utf-8

# I wanted to try to learn how the YouTube trending page algorithm works. I found this dataset on most viewed videos of 2018. 
# > * Top 10 most watched videos of 2018
# > * Comparing views vs likes, dislikes and number of comments

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

videos_usa = pd.read_csv("../input/USvideos.csv")
videos_gb = pd.read_csv("../input/GBvideos.csv")


# In[ ]:


videos_usa.head()


# In[ ]:


videos_usa = videos_usa.drop_duplicates(subset='title',keep='first')


# In[ ]:


videos_usa.sort_values(by='views',ascending=False)


# **Top 10 most watched videos of 2018**

# In[ ]:


videos_usa_top10 = videos_usa.nlargest(10, 'views')


# In[ ]:


videos_usa_top10.head()


# In[ ]:


plt.figure(figsize=(13, 7), dpi=200)
plt.bar(videos_usa_top10['title'],videos_usa_top10['views'])
plt.xticks(rotation=90)
plt.ylabel('views')
plt.title('Number of views of top 10 videos')
plt.show()


# **Comparing views vs likes, dislikes and number of comments**

# In[ ]:


likes = sns.jointplot(x ='views', y ='likes', data=videos_usa, kind='reg')
likes.annotate(stats.pearsonr)
comment_count = sns.jointplot(x ='views', y ='comment_count', data=videos_usa, kind='reg')
comment_count.annotate(stats.pearsonr)
dislikes = sns.jointplot(x ='views', y ='dislikes', data=videos_usa, kind='reg')
dislikes.annotate(stats.pearsonr)
plt.figure(figsize=(16, 9), dpi=200)
plt.show()


# In[ ]:




