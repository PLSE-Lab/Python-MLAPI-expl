#!/usr/bin/env python
# coding: utf-8

# ## What are recommendation Engines?

# Till recently, people generally tended to buy products recommended to them by their friends or the people they trust. This used to be the primary method of purchase when there was any doubt about the product. But with the advent of the digital age, that circle has expanded to include online sites that utilize some sort of recommendation engine.
# A recommendation engine filters the data using different algorithms and recommends the most relevant items to users. It first captures the past behavior of a customer and based on that, recommends products which the users might be likely to buy.
# 
# **A recommendation engine filters the data using different algorithms and recommends the most relevant items to users. It first captures the past behavior of a customer and based on that, recommends products which the users might be likely to buy.**
# 
# 
# The rapid growth of data collection has led to a new era of information. Data is being used to create more efficient systems and this is where Recommendation Systems come into play. Recommendation Systems are a type of information filtering systems as they improve the quality of search results and provides items that are more relevant to the search item or are realted to the search history of the user.
# 
# They are used to predict the rating or preference that a user would give to an item. Almost every major tech company has applied them in some form or the other: Amazon uses it to suggest products to customers, YouTube uses it to decide which video to play next on autoplay, and Facebook uses it to recommend pages to like and people to follow. Moreover, companies like Netflix and Spotify depend highly on the effectiveness of their recommendation engines for their business and sucees.
# 
# ![Screenshot_2020-01-10%20recommendation%20engine%20youtube%20-%20Google%20Search.png](attachment:Screenshot_2020-01-10%20recommendation%20engine%20youtube%20-%20Google%20Search.png)

# In[ ]:


# analysing the Dataset
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Importing required libraries

# In[ ]:


from IPython.core.interactiveshell import InteractiveShell 
InteractiveShell.ast_node_interactivity = "all"


# ## Exploratary Data Analysis

# In[ ]:


import pandas as pd
import pandas_profiling 


# In[ ]:


# Reading the data-file
df = pd.read_csv('/kaggle/input/location-visited-by-travller-in-india-fake/dataset.csv')


# In[ ]:


df.head()


# ### Pandas Profiling
# 
# Profiling is a process that helps us in understanding our data and [PandasProfiling](https://github.com/pandas-profiling/pandas-profiling) is python package which does exactly that. It is a simple and fast way to perform exploratory data analysis of a Pandas Dataframe.

# In[ ]:


df.profile_report()


# In[ ]:


# no of unique users in our dataset
df['Name'].nunique()


# In[ ]:


# no of unique city
df['City'].nunique()


# In[ ]:


# no of unique country
df['Country'].unique()


# In[ ]:


import seaborn as sns


# In[ ]:


# Analysing user preference w.r.t gender
sns.countplot(x='prefeerences', hue='gender', data=df)


# In[ ]:


# counting no of cities
sns.countplot(x='City', data=df)


# ## Types of Recommendation Engine
# 
# There are basically three types of recommender systems:-
# 
# - **Demographic Filtering**- They offer generalized recommendations to every user, based on movie popularity and/or genre. The System recommends the same movies to users with similar demographic features. Since each user is different , this approach is considered to be too simple. The basic idea behind this system is that movies that are more popular and critically acclaimed will have a higher probability of being liked by the average audience.
# - **Content Based Filtering-** They suggest similar items based on a particular item. This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations. The general idea behind these recommender systems is that if a person liked a particular item, he or she will also like an item that is similar to it.
# - **Collaborative Filtering**- This system matches persons with similar interests and provides recommendations based on this matching. Collaborative filters do not require item metadata like its content-based counterparts.

# In[ ]:


# installing recommendation engine python package
get_ipython().system(' pip install turicreate')


# In[ ]:


# read dataset
import turicreate as tc
actions = tc.SFrame.read_csv('/kaggle/input/location-visited-by-travller-in-india-fake/dataset.csv')


# ### Goal
# 
# **For a given user recommend the city to travel next for a vacation in India**
# ![rome-woman_3209572a.jpg](attachment:rome-woman_3209572a.jpg)

# In[ ]:


# Create a recommendation engine for a Particular user with name and engine
model = tc.recommender.create(actions, 'Name', 'City')


# In[ ]:


## Recommended cities
model.recommend()


# ### References
# 
# - https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system
# - https://apple.github.io/turicreate/docs/userguide/recommender/
# - https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/
# - https://www.kaggle.com/parulpandey/10-simple-hacks-to-speed-up-your-data-analysis
