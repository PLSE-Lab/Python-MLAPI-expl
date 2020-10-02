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
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# ### Loading and Reading Dataset

# In[ ]:


ratings  = pd.read_csv('../input/ratings_small.csv')
ratings.head()


# In[ ]:


print(ratings.shape)


# ## Splitting Data into Train and Test Set

# In[ ]:


from sklearn.model_selection import train_test_split
train_df,test_df = train_test_split(ratings, test_size = 0.3, random_state = 42)
print(train_df.shape, '\t\t', test_df.shape)


# In[ ]:


train_df.head()


# #### As MovieID column stores, movies and our first task is to build a recommendation engine based on USER COLLABORATIVE approach, we want our data in tabular format such that: userID as index, Data i.e Unique MovieID's as Columns/features and Ratings as Values

# ### We can achieve this using dataframe's Pivot Method

# In[ ]:


df_movies_as_features = train_df.pivot(index = 'userId', columns = 'movieId',values = 'rating' )
df_movies_as_features.shape


# In[ ]:


df_movies_as_features.head()


# In[ ]:


df_movies_as_features.fillna(0, inplace = True)
df_movies_as_features.head()


# In[ ]:





#   ####  Copy Train and Test DATASET
#   This will be used for Evaluation and

# ### Let's create User Similarity Matrix
# Using Cosine Similarities

# 

# In[ ]:


from sklearn.metrics.pairwise import pairwise_distances

