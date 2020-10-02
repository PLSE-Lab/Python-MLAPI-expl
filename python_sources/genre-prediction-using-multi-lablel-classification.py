#!/usr/bin/env python
# coding: utf-8

# #### Introduction 
# In this notebook I will try to develop a multi-label genre classifier using given data of movie plots.

# In[1]:


import pandas as pd 


# In[2]:


movie_plots = pd.read_csv('../input/wiki_movie_plots_deduped.csv')


# In[3]:


movie_plots.head()


# #### Exploratory Data Analysis

# In[4]:


movie_plots.shape


# In[13]:


movie_plots.isnull().sum()


# In[11]:


#Total unique combinations of genres

len(df['Genre'].unique())


# In[8]:


#Checking most popular genres

df = movie_plots.groupby(["Genre"]).size().reset_index(name='count')
df = df.sort_values(by=['count'], ascending=False)
df.head(n=50)


# In[14]:


#Keeping only generes with 50+ instances

df = df[df['count']>=50]


# In[15]:


df.shape


# In[18]:


genre_list = df['Genre'].tolist()
genre_list


# In[21]:


#Removing genre 'unknown'

genre_list = genre_list[1:]


# In[22]:


movie_plots_ = movie_plots[movie_plots['Genre'].isin(genre_list)]


# #### Feature Analysis
# Though I will be building classifier using movie plots only lets also alnalyse other features available in this dataset

# In[25]:


len(movie_plots_['Origin/Ethnicity'].unique())


# _Work in progress, will be adding further analysis soon_

# In[ ]:




