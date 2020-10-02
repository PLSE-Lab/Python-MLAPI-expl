#!/usr/bin/env python
# coding: utf-8

# # 0.Preface

# Perform some EDA on the Apple store data

# # 1.Data loading

# ## 1.1 Loaded needed package into memory

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        


# ## 1.2 Loaded the needed data

# In[ ]:


path = '/kaggle/input/app-store-apple-data-set-10k-apps'
os.chdir(path)
app_data = pd.read_csv('AppleStore.csv')
app_data.head()


# List some stastics to get a basic understanding about data set.

# In[ ]:


app_data.describe().T


# # 2.Exploration

# ## 2.1.1 What are the Top 10 App from user rating perspective

# In[ ]:


app_data = pd.read_csv('AppleStore.csv')
app_data_rating = app_data.sort_values('user_rating',ascending=False)[['track_name','user_rating','price']]
app_data_rating[:10]


# It seems that I need to construct a new variable to describe the rating of a App, since there are lot's of Appp is with the same score, which is 5.
# But you can see that most of the App that is with score 5 are for free.

# ## 2.1.2 What are the Bottom 10 App from user rating perspective

# In[ ]:


app_data = pd.read_csv('AppleStore.csv')
app_data_rating = app_data.sort_values('user_rating',ascending=False)[['track_name','user_rating','price']]
app_data_rating[-10:]


# ## 2.2.1 What are the Top 10 App from user price perspective

# In[ ]:


app_data_price = app_data.sort_values('price',ascending=False)[['track_name','user_rating','price']]
app_data_price[:10]


# You can find that the top one expensive App is 299.99 USD, and there is no App that is score 5

# ## 2.2.3 Price distribution

# Check the distribution of price by using hist plot, but I found it is really hard to tell something from it, it seems because there are so many Apps that are with low price, or even 0 USD, basiclly I can say that the data is with a extreamly high peak and a short tail on the right.

# In[ ]:


app_data_price_hist = app_data[['price']]
app_data_price_hist.plot.hist(bins=100)


# ## 2.2.4 Rating distribution

# Comparing with price, rating distribution is queit clear, interestingly the greater part of the App are rated with score 4.5-5.0

# In[ ]:


app_data_rating_hist = app_data[['user_rating']]
app_data_rating_hist.plot.hist(bins=10)


# ## 2.2.5 Number of category

# Let's see how many apps in each category, we can find that the most popular category is game, there are 3862 Apps that are game, and there are 535 Apps that are entertainment.

# In[ ]:


app_data_cate_bar = app_data.groupby(['prime_genre'])[['id']].count().reset_index().sort_values('id',ascending=False)
app_data_cate_bar.columns = ['prime_genre','Nbr']
top_categories = app_data_cate_bar.head(10)

sns.barplot(y = 'prime_genre',x = 'Nbr', data=top_categories)
top_categories


# And we can also find what are the unpopular categories in Apple store, you can see that there are only 10 Apps that is "Catalogs" type. I don't know what is "Catalog" type of App,what are they used for? is there anyone knows?

# In[ ]:


top_categories = app_data_cate_bar.tail(10)
sns.barplot(y = 'prime_genre',x = 'Nbr', data=top_categories)
top_categories

