#!/usr/bin/env python
# coding: utf-8

# ![](data:image/jpeg;base64,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)

# # Introduction
# 
# **Hello Kagglers! Welcome to the my second project.** 
# 
# **In this project I will explore:**
# 1. [Top Free Games,](#1)
# 1. [Top Paid Games,](#2)
# 1. [Top Free Apps,](#3)
# 1. [Top Paid Apps,](#4)
# 1. [Top Trending Free Games,](#5)
# 1. [Top Trending Paid Games,](#6)
# 1. [Top Trending Free Apps,](#7)
# 1. [Top Trending Paid Apps,](#8)
# in **Apple Store** using python and **mostly pandas** library.
# 
# 
# P.S.
# All these results explored according to our dataset. So, real statistics might be different than our results. Surely, Apple uses more complex algorithms.
# 
# 

# >  Please leave me a comment and upvote the kernel if you liked at the end.

# **Basic Imports**

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# **Getting The Data**

# In[ ]:


df = pd.read_csv("../input/AppleStore.csv")


# In[ ]:


df.head()


# In[ ]:


df.info()


# * According to dataset, we can explore the data on 2 ways. By using current version and ratings of apps named as **"trend apps"** and by using total ratings of apps named as **"best apps"**.

# **Lets start from best ones!**
# * Wait! Firstly, clean the data.

# In[ ]:


ratings = df.loc[:,["track_name","prime_genre","user_rating","rating_count_tot","price"]]
ratings = ratings.sort_values(by=["user_rating","rating_count_tot"],ascending=False)
ratings.head()


# **Plotting**

# In[ ]:


plt.figure(figsize=(10,5))
sns.countplot(y= ratings["prime_genre"])


# * **According to chart AppStore has lot of games. So, we should noticed that we will classify the data on 2 ways, games and apps.**

# In[ ]:


sns.countplot(ratings["price"]==0)


# * **According to that chart AppStore has lot of free games and apps. So, we should noticed that we will classify the price on 2 ways, free and paid.**

# In[ ]:


plt.figure(figsize=(10,4))
sns.countplot(ratings["user_rating"])


# * **According to that chart, approximately 1000 apps did not voted by users. These apps might me new and they can block us to find correct results. Therefore, i will work on apps which voted more than average of all apps.**

# In[ ]:


ratings["rating_count_tot"].mean()


# In[ ]:


ratings = ratings[ratings["rating_count_tot"]>ratings["rating_count_tot"].mean()]


# In[ ]:


plt.figure(figsize=(10,4))
sns.countplot(ratings["user_rating"])


# *** Currently, we cleaned our data from mostly unvoted new apps.**

# # Then Let's Explore Now!

# <a id="1"></a> <br>
# # Top Free Games

# In[ ]:


top_free_games = ratings[(ratings["prime_genre"]=="Games") & (ratings["price"]==0)]
top_free_games.head(10)


# <a id="2"></a> <br>
# # Top Paid Games

# In[ ]:


top_paid_games = ratings[(ratings["prime_genre"]=="Games") & (ratings["price"]!=0)]
top_paid_games.head(10)


# <a id="3"></a> <br>
# # Top Free Apps

# In[ ]:


top_free_apps = ratings[(ratings["prime_genre"]!="Games") & (ratings["price"]==0)]
top_free_apps.head(10)


# <a id="4"></a> <br>
# # Top Paid Apps

# In[ ]:


top_paid_apps = ratings[(ratings["prime_genre"]!="Games") & (ratings["price"]!=0)]
top_paid_apps.head(10)


# * **Well, we found bests of all time but what about trendings? For that, we will use data columns of latest version and will repeat the first steps.**

# In[ ]:


trend_ratings = df.loc[:,["track_name","prime_genre","user_rating_ver","rating_count_ver","price"]]


# In[ ]:


trend_ratings = trend_ratings[trend_ratings["rating_count_ver"]>trend_ratings["rating_count_ver"].mean()]


# In[ ]:


plt.figure(figsize=(10,4))
sns.countplot(trend_ratings["user_rating_ver"])


# In[ ]:


trend_ratings = trend_ratings.sort_values(by=["user_rating_ver","rating_count_ver"],ascending=False)


# In[ ]:


trend_ratings.head()


#  **We cleaned our data and lets begin!**

# <a id="5"></a> <br>
# # Top Trending Free Games

# In[ ]:


top_trending_free_games = trend_ratings[(trend_ratings["prime_genre"]=="Games") & (trend_ratings["price"]==0)]
top_trending_free_games.head(10)


# <a id="6"></a> <br>
# # Top Trending Paid Games

# In[ ]:


top_trending_paid_games = trend_ratings[(trend_ratings["prime_genre"]=="Games") & (trend_ratings["price"]!=0)]
top_trending_paid_games.head(10)


# <a id="7"></a> <br>
# # Top Trending Free Apps

# In[ ]:


top_trending_free_apps = trend_ratings[(trend_ratings["prime_genre"]!="Games") & (trend_ratings["price"]==0)]
top_trending_free_apps.head(10)


# <a id="8"></a> <br>
# # Top Trending Paid Apps

# In[ ]:


top_trending_paid_apps = trend_ratings[(trend_ratings["prime_genre"]!="Games") & (trend_ratings["price"]!=0)]
top_trending_paid_apps.head(10)


# # Thank you for your time and attention!

# >  Please leave me a comment and upvote the kernel if you liked.
